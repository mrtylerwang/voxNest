//! Tauri Commands
//! 前端可调用的 Rust 命令

use tauri::{AppHandle, Manager, Runtime};
use tauri::menu::{Menu, MenuItem, Submenu, PredefinedMenuItem};
use crate::python::PythonBackend;
use base64::{engine::general_purpose::STANDARD, Engine as _};

/// 打开外部链接（在默认浏览器中）
#[tauri::command]
pub fn open_external(url: String) -> Result<(), String> {
    // 验证 URL 格式
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Err("无效的 URL 格式".to_string());
    }

    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(&url)
            .spawn()
            .map_err(|e| format!("无法打开链接: {}", e))?;
    }

    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("cmd")
            .args(&["/C", "start", "", &url])
            .spawn()
            .map_err(|e| format!("无法打开链接: {}", e))?;
    }

    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open")
            .arg(&url)
            .spawn()
            .map_err(|e| format!("无法打开链接: {}", e))?;
    }

    Ok(())
}

/// 获取后端服务端口
#[tauri::command]
pub fn get_backend_port<R: Runtime>(app: AppHandle<R>) -> Result<Option<u16>, String> {
    let backend = app.state::<PythonBackend>();
    Ok(backend.get_port())
}

/// 手动重启后端服务
#[tauri::command]
pub async fn restart_backend<R: Runtime>(app: AppHandle<R>) -> Result<u16, String> {
    let backend = app.state::<PythonBackend>();

    // 停止现有服务
    backend.stop().map_err(|e| e.to_string())?;

    // 等待一下
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // 启动新服务（重启时不显示启动进度画面）
    let port = backend.start(&app, None).map_err(|e| e.to_string())?;

    Ok(port)
}

/// 获取应用版本
#[tauri::command]
pub fn get_app_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// 保存文本文件到指定路径（用于 SRT 导出、日志导出等）
#[tauri::command]
pub fn save_file(path: String, content: String) -> Result<(), String> {
    std::fs::write(&path, content)
        .map_err(|e| format!("保存文件失败: {}", e))
}

/// 保存 Base64 编码的二进制文件（用于音频另存等）
#[tauri::command]
pub fn save_binary_file(path: String, base64_data: String) -> Result<(), String> {
    let payload = base64_data
        .split_once(',')
        .map(|(_, data)| data)
        .unwrap_or(base64_data.as_str());

    let bytes = STANDARD
        .decode(payload)
        .map_err(|e| format!("解码文件数据失败: {}", e))?;

    std::fs::write(&path, bytes)
        .map_err(|e| format!("保存文件失败: {}", e))
}

/// 根据当前语言重建 macOS 系统菜单栏
#[tauri::command]
pub fn rebuild_menu<R: Runtime>(app: AppHandle<R>, lang: String) -> Result<(), String> {
    let menu = build_menu(&app, &lang).map_err(|e| e.to_string())?;
    app.set_menu(menu).map_err(|e| e.to_string())?;
    // 窗口标题同步跟随语言：macOS 26+ 的 Dock 右键菜单首行显示
    // 活动窗口标题（带 ✓），必须与菜单语言一起更新才能正确显示品牌名
    let zh = lang.starts_with("zh");
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.set_title(if zh { "声巢" } else { "voxNest" });
    }
    Ok(())
}

pub fn build_menu<R: Runtime>(app: &AppHandle<R>, lang: &str) -> tauri::Result<Menu<R>> {
    let zh = lang.starts_with("zh");
    fn pick<'a>(zh: bool, zh_cn: &'a str, en: &'a str) -> &'a str {
        if zh { zh_cn } else { en }
    }

    // 应用菜单（中文侧品牌名用「声巢」，英文侧用 voxNest）
    let about = MenuItem::with_id(
        app, "about", pick(zh, "关于声巢", "About voxNest"), true, None::<&str>
    )?;
    let hide = PredefinedMenuItem::hide(app, Some(pick(zh, "隐藏声巢", "Hide voxNest")))?;
    let hide_others = PredefinedMenuItem::hide_others(app, Some(pick(zh, "隐藏其他", "Hide Others")))?;
    let quit = PredefinedMenuItem::quit(app, Some(pick(zh, "退出声巢", "Quit voxNest")))?;
    let app_menu = Submenu::new(app, pick(zh, "声巢", "voxNest"), true)?;
    app_menu.append(&about)?;
    app_menu.append(&PredefinedMenuItem::separator(app)?)?;
    app_menu.append(&hide)?;
    app_menu.append(&hide_others)?;
    let show_all = PredefinedMenuItem::show_all(app, Some(pick(zh, "显示全部", "Show All")))?;
    app_menu.append(&show_all)?;
    app_menu.append(&PredefinedMenuItem::separator(app)?)?;
    app_menu.append(&quit)?;

    // 文件菜单
    let close = PredefinedMenuItem::close_window(app, Some(pick(zh, "关闭窗口", "Close Window")))?;
    let file_menu = Submenu::new(app, pick(zh, "文件", "File"), true)?;
    file_menu.append(&close)?;

    // 编辑菜单（copy 用 Apple 官方中文用词「拷贝」）
    let edit_menu = Submenu::new(app, pick(zh, "编辑", "Edit"), true)?;
    edit_menu.append(&PredefinedMenuItem::undo(app, Some(pick(zh, "撤销", "Undo")))?)?;
    edit_menu.append(&PredefinedMenuItem::redo(app, Some(pick(zh, "重做", "Redo")))?)?;
    edit_menu.append(&PredefinedMenuItem::separator(app)?)?;
    edit_menu.append(&PredefinedMenuItem::cut(app, Some(pick(zh, "剪切", "Cut")))?)?;
    edit_menu.append(&PredefinedMenuItem::copy(app, Some(pick(zh, "拷贝", "Copy")))?)?;
    edit_menu.append(&PredefinedMenuItem::paste(app, Some(pick(zh, "粘贴", "Paste")))?)?;
    edit_menu.append(&PredefinedMenuItem::select_all(app, Some(pick(zh, "全选", "Select All")))?)?;

    // 视图菜单
    let view_menu = Submenu::new(app, pick(zh, "显示", "View"), true)?;
    let fullscreen_item = PredefinedMenuItem::fullscreen(app, Some(pick(zh, "全屏", "Enter Full Screen")))?;
    view_menu.append(&fullscreen_item)?;

    // 窗口菜单
    let win_menu = Submenu::new(app, pick(zh, "窗口", "Window"), true)?;
    let min_item = PredefinedMenuItem::minimize(app, Some(pick(zh, "最小化", "Minimize")))?;
    win_menu.append(&min_item)?;
    let max_item = PredefinedMenuItem::maximize(app, Some(pick(zh, "缩放", "Zoom")))?;
    win_menu.append(&max_item)?;

    // 帮助菜单
    let help_menu = Submenu::new(app, pick(zh, "帮助", "Help"), true)?;
    let docs = MenuItem::with_id(
        app, "docs", pick(zh, "声巢帮助", "voxNest Help"), true, None::<&str>
    )?;
    help_menu.append(&docs)?;

    let menu = Menu::new(app)?;
    menu.append(&app_menu)?;
    menu.append(&file_menu)?;
    menu.append(&edit_menu)?;
    menu.append(&view_menu)?;
    menu.append(&win_menu)?;
    menu.append(&help_menu)?;
    Ok(menu)
}
