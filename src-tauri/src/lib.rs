//! VoxNest Tauri 应用主入口
//! 负责 Python 后端服务的启动、端口注入、窗口管理

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod port;
mod python;
mod commands;

use tauri::{Manager, Emitter};
use python::PythonBackend;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            println!("[VoxNest] 应用启动中...");

            // 初始化 Python 后端管理器
            let backend = PythonBackend::new();
            app.manage(backend);

            // 构建系统菜单栏（初始按系统语言）
            let initial_lang = if std::env::var("LANG").unwrap_or_default().starts_with("zh") { "zh" } else { "en" };
            if let Ok(menu) = crate::commands::build_menu(&app.handle(), initial_lang) {
                let _ = app.handle().set_menu(menu);
            }

            // 菜单点击事件：about / docs
            let app_handle_for_menu = app.handle().clone();
            app.on_menu_event(move |_window, event| {
                let id = event.id().0.as_str();
                match id {
                    "about" => { let _ = app_handle_for_menu.emit("voxnest://menu-about", ()); }
                    "docs" => { let _ = app_handle_for_menu.emit("voxnest://menu-docs", ()); }
                    _ => {}
                }
            });

            // 先创建主窗口（显示启动画面）
            let handle = app.handle().clone();
            let window = tauri::WebviewWindowBuilder::new(
                &handle,
                "main",
                tauri::WebviewUrl::App("index.html".into()),
            )
            // 窗口标题必须显式设置：Tauri 2 默认标题为 "Tauri App"，
            // macOS 26+ 的 Dock 右键菜单首行显示活动窗口标题而非应用名。
            // 标题跟随初始系统语言（品牌双语策略：zh=声巢 / en=voxNest），
            // 运行中语言切换由 rebuild_menu 命令同步 set_title。
            .title(if initial_lang == "zh" { "声巢" } else { "voxNest" })
            .initialization_script(
                r#"
                    window.__VOXNEST_PORTS__ = { backend: 0 };
                    window.__VOXNEST_STARTING__ = true;
                "#
            )
            .inner_size(1280.0, 820.0)
            .min_inner_size(980.0, 660.0)
            .resizable(true)
            .fullscreen(false)
            .title_bar_style(tauri::TitleBarStyle::Overlay)
            .hidden_title(true)
            .build()
            .map_err(|e| {
                eprintln!("[VoxNest] 创建主窗口失败: {}", e);
                e
            })?;

            println!("[VoxNest] 主窗口已创建，开始异步启动后端...");

            // 异步启动后端（不阻塞窗口显示）
            let app_handle = app.handle().clone();
            let window_for_progress = window.clone();
            let window_for_complete = window.clone();
            std::thread::spawn(move || {
                // 进度回调：通过 Tauri 事件发送到前端
                let progress_cb: Box<dyn Fn(&str, u8, &str) + Send> = Box::new(
                    move |stage: &str, percent: u8, message: &str| {
                        println!("[VoxNest] 启动进度 [{}%] {}: {}", percent, stage, message);
                        let _ = window_for_progress.emit(
                            "voxnest://startup-progress",
                            serde_json::json!({
                                "stage": stage,
                                "percent": percent,
                                "message": message
                            }),
                        );
                    },
                );

                // 获取后端管理器并启动
                if let Some(backend) = app_handle.try_state::<PythonBackend>() {
                    match backend.start(&app_handle, Some(progress_cb)) {
                        Ok(port) => {
                            println!("[VoxNest] Python 后端已启动，端口: {}", port);
                            // 注入端口到前端
                            match window_for_complete.eval(&format!(
                                "window.__VOXNEST_PORTS__ = {{ backend: {} }}; window.__VOXNEST_STARTING__ = false;",
                                port
                            )) {
                                Ok(_) => println!("[VoxNest] eval 注入端口成功: {}", port),
                                Err(e) => eprintln!("[VoxNest] eval 注入端口失败: {}", e),
                            }
                            match window_for_complete.emit("voxnest://startup-complete", serde_json::json!({ "port": port })) {
                                Ok(_) => println!("[VoxNest] startup-complete 事件已发送"),
                                Err(e) => eprintln!("[VoxNest] startup-complete 事件发送失败: {}", e),
                            }
                        }
                        Err(e) => {
                            eprintln!("[VoxNest] Python 后端启动失败: {}", e);
                            let _ = window_for_complete.emit(
                                "voxnest://startup-error",
                                serde_json::json!({ "error": e.to_string() }),
                            );
                        }
                    }
                }
            });

            println!("[VoxNest] 应用启动完成（后端异步启动中）");
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::open_external,
            commands::get_backend_port,
            commands::restart_backend,
            commands::get_app_version,
            commands::save_file,
            commands::save_binary_file,
            commands::rebuild_menu,
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                println!("[VoxNest] 窗口关闭，正在停止后端服务并退出应用...");
                let app_handle = window.app_handle().clone();
                // 停止后端服务
                if let Some(backend) = app_handle.try_state::<PythonBackend>() {
                    let _ = backend.stop();
                }
                // 兜底：pkill 杀死所有 uvicorn/python 后端进程（防止残留）
                // 注意：禁止 pkill -f voxnest（AGENTS.md）——会误杀应用自身
                let _ = std::process::Command::new("pkill")
                    .args(&["-f", "uvicorn app.main:app"])
                    .spawn();
                // 退出应用（关闭窗口后完全退出，不在后台运行）
                std::thread::spawn(move || {
                    std::thread::sleep(std::time::Duration::from_millis(800));
                    app_handle.exit(0);
                });
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
