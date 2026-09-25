//! Python 后端子进程管理
//! 负责启动、监控、停止 Python FastAPI 后端服务

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::path::PathBuf;
use std::sync::Mutex;
use tauri::{AppHandle, Manager, Runtime};
use anyhow::{Result, anyhow};
#[cfg(unix)]
use std::os::unix::process::CommandExt;

use crate::port::find_free_port;

/// Python 后端进程管理器
pub struct PythonBackend {
    child: Mutex<Option<Child>>,
    port: Mutex<Option<u16>>,
}

impl PythonBackend {
    pub fn new() -> Self {
        Self {
            child: Mutex::new(None),
            port: Mutex::new(None),
        }
    }

    /// 启动 Python 后端服务
    /// progress_callback: 启动进度回调（阶段, 百分比0-100, 消息）
    pub fn start<R: Runtime>(
        &self,
        app: &AppHandle<R>,
        progress_callback: Option<Box<dyn Fn(&str, u8, &str) + Send>>,
    ) -> Result<u16> {
        // 检查是否已在运行
        {
            let child_guard = self.child.lock().unwrap();
            if child_guard.is_some() {
                // 已在运行，返回现有端口
                if let Some(port) = *self.port.lock().unwrap() {
                    return Ok(port);
                }
            }
        }

        // 查找可用端口
        let port = find_free_port().ok_or_else(|| anyhow!("无法找到可用端口"))?;

        if let Some(ref cb) = progress_callback {
            cb("locating", 5, "正在定位 Python 运行时...");
        }

        // 解析最终 resource_dir（含 sandbox 路径隔离时的 exe 推导回退），贯穿整个启动链
        let resource_dir = self.resolve_resource_dir(app)?;

        // 定位 Python 解释器和后端代码
        let (python_path, backend_path) = self.locate_python_and_backend(app, &resource_dir)?;

        println!("[PythonBackend] Python: {:?}", python_path);
        println!("[PythonBackend] Backend: {:?}", backend_path);
        println!("[PythonBackend] Port: {}", port);

        // 打包模式下：首次启动自动创建 venv 并安装依赖
        // 确保用户无需安装 Python，所有依赖内置
        let python_path = self.ensure_venv(app, &resource_dir, &python_path, &backend_path, progress_callback.as_ref())?;

        if let Some(ref cb) = progress_callback {
            cb("starting", 95, "正在启动后端服务...");
        }

        // 构建启动命令
        // 使用 Stdio::piped 并起 drain 线程：既不吞 uvicorn 错误，又避免管道写满阻塞
        let mut command = Command::new(&python_path);
        command
            .arg("-m")
            .arg("uvicorn")
            .arg("app.main:app")
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("--log-level")
            .arg("info")
            .current_dir(&backend_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // 注入环境变量
        command.env("PYTHONUNBUFFERED", "1");
        command.env("PYTHONDONTWRITEBYTECODE", "1");

        // 注入 ffmpeg PATH（AGENTS.md 强制要求：Tauri 子进程 PATH 必须包含 /opt/homebrew/bin）
        // 优先使用应用内置 ffmpeg（resource_dir/bin），其次 homebrew，最后系统 PATH
        let mut ffmpath = String::new();
        {
            let bin_dir = resource_dir.join("bin");
            if bin_dir.exists() {
                ffmpath.push_str(&bin_dir.to_string_lossy());
                ffmpath.push(':');
            }
        }
        // macOS Apple Silicon: /opt/homebrew/bin, Intel: /usr/local/bin
        if cfg!(target_os = "macos") {
            ffmpath.push_str("/opt/homebrew/bin:/usr/local/bin:");
        }
        if let Ok(existing_path) = std::env::var("PATH") {
            ffmpath.push_str(&existing_path);
        }
        command.env("PATH", &ffmpath);

        // 设置资源目录（后端用于查找内置 ffmpeg 等资源），使用解析后的 resource_dir
        command.env("VOXNEST_RESOURCE_DIR", resource_dir.to_string_lossy().to_string());
        println!("[PythonBackend] VOXNEST_RESOURCE_DIR: {:?}", resource_dir);
        // 开发模式下也设置 tauri-app 根目录（用于查找 tauri-app/bin/ffmpeg）
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                let dir_str = exe_dir.to_string_lossy();
                if dir_str.contains("target/debug") || dir_str.contains("target/release") {
                    // 推导 tauri-app 目录：target/debug/ -> 上溯3层
                    if let Some(project_root) = exe_dir.ancestors().nth(3) {
                        command.env("VOXNEST_PROJECT_ROOT", project_root.to_string_lossy().to_string());
                    }
                }
            }
        }

        #[cfg(unix)]
        unsafe {
            command.pre_exec(|| {
                if libc::setsid() == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }

        // 启动子进程
        let mut child = command.spawn()
            .map_err(|e| anyhow!("启动 Python 后端失败: {}", e))?;

        // 起 drain 线程持续消费 stdout/stderr，防止管道写满阻塞（AGENTS.md 强制要求）
        // 所有 uvicorn 日志都透传到 Rust println!/eprintln!，方便定位崩溃
        if let Some(stdout) = child.stdout.take() {
            std::thread::spawn(move || {
                let reader = BufReader::new(stdout);
                for line in reader.lines().flatten() {
                    println!("[uvicorn] {}", line);
                }
            });
        }
        if let Some(stderr) = child.stderr.take() {
            std::thread::spawn(move || {
                let reader = BufReader::new(stderr);
                for line in reader.lines().flatten() {
                    eprintln!("[uvicorn-err] {}", line);
                }
            });
        }

        // 保存进程和端口
        *self.child.lock().unwrap() = Some(child);
        *self.port.lock().unwrap() = Some(port);

        println!("[PythonBackend] 后端服务已启动，PID: {:?}",
            self.child.lock().unwrap().as_ref().unwrap().id());

        Ok(port)
    }

    /// 停止 Python 后端服务
    pub fn stop(&self) -> Result<()> {
        let mut child_guard = self.child.lock().unwrap();
        if let Some(mut child) = child_guard.take() {
            let pid = child.id();
            println!("[PythonBackend] 正在停止后端服务，PID: {}", pid);

            #[cfg(not(target_os = "windows"))]
            {
                // 第一步：发送 SIGTERM（优雅停止）
                unsafe {
                    // 同时杀死整个进程组（uvicorn 可能有子进程）
                    libc::killpg(pid as i32, libc::SIGTERM);
                }

                // 等待进程退出（最多 3 秒）
                let start = std::time::Instant::now();
                loop {
                    match child.try_wait() {
                        Ok(Some(_)) => {
                            println!("[PythonBackend] 后端服务已优雅停止");
                            break;
                        }
                        Ok(None) => {
                            if start.elapsed().as_secs() >= 3 {
                                // 超时，发送 SIGKILL 强制杀死
                                println!("[PythonBackend] 优雅停止超时，发送 SIGKILL 强制杀死");
                                unsafe {
                                    libc::killpg(pid as i32, libc::SIGKILL);
                                }
                                let _ = child.wait();
                                println!("[PythonBackend] 后端服务已强制停止");
                                break;
                            }
                            std::thread::sleep(std::time::Duration::from_millis(100));
                        }
                        Err(_) => {
                            println!("[PythonBackend] 等待进程退出失败");
                            break;
                        }
                    }
                }

                // 如果进程组未能完全回收，再尝试补一个主进程 kill。
                let _ = child.kill();
            }

            #[cfg(target_os = "windows")]
            {
                use std::process::Command;
                let _ = Command::new("taskkill")
                    .args(&["/PID", &pid.to_string(), "/F", "/T"])
                    .spawn();
                let _ = child.wait();
            }
        }

        *self.port.lock().unwrap() = None;
        Ok(())
    }

    /// 获取当前端口
    pub fn get_port(&self) -> Option<u16> {
        *self.port.lock().unwrap()
    }

    /// 检查后端是否在运行
    #[allow(dead_code)]
    pub fn is_running(&self) -> bool {
        self.child.lock().unwrap().is_some()
    }

    /// 定位 Python 解释器和后端代码路径（resource_dir 已由 resolve_resource_dir 解析）
    fn locate_python_and_backend<R: Runtime>(&self, app: &AppHandle<R>, resource_dir: &PathBuf) -> Result<(PathBuf, PathBuf)> {
        // 0. 环境变量优先（用于开发测试）
        if let Ok(python_path) = std::env::var("VOXNEST_PYTHON") {
            let python = PathBuf::from(&python_path);
            if python.exists() {
                let backend_dir = std::env::var("VOXNEST_BACKEND")
                    .map(PathBuf::from)
                    .unwrap_or_else(|_| PathBuf::from("."));
                if backend_dir.exists() {
                    println!("[PythonBackend] 使用环境变量指定的 Python 和 Backend");
                    return Ok((python, backend_dir));
                }
            }
        }

        // 1. 开发环境优先：通过可执行文件路径推导项目根目录
        //    dev 模式下可执行文件在 tauri-app/src-tauri/target/debug/ 或 release/
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()));

        if let Some(dir) = exe_dir {
            // 判断是否为开发模式（路径包含 target/debug 或 target/release）
            let is_dev = dir.to_string_lossy().contains("target/debug")
                || dir.to_string_lossy().contains("target/release");

            if is_dev {
                // 尝试多种相对路径找到 tauri-app/backend（目录已收敛到 tauri-app 内）
                // target/debug/ 上溯 3 层即 tauri-app/
                let candidates = vec![
                    dir.join("../../../backend"),            // target/debug/ -> tauri-app/backend
                    dir.join("../../backend"),               // target/ -> tauri-app/backend (少一层时)
                    dir.join("../../../../tauri-app/backend"), // 兜底: 从仓库根推导
                ];

                for backend_dir in candidates {
                    if backend_dir.exists() && backend_dir.join("app").exists() {
                        // 开发环境使用项目 venv（backend/venv 或项目根 .venv）
                        let venv_candidates = vec![
                            backend_dir.join("venv").join("bin").join("python"),
                            backend_dir.parent()
                                .map(|p| p.join(".venv").join("bin").join("python"))
                                .unwrap_or_else(|| PathBuf::new()),
                        ];

                        for python in venv_candidates {
                            if python.exists() {
                                println!("[PythonBackend] 开发模式: 使用项目 venv");
                                return Ok((python, backend_dir));
                            }
                        }

                        // 找到 backend 但没找到 venv
                        println!("[PythonBackend] 警告: 找到 backend 但未找到 venv: {:?}", backend_dir);
                    }
                }
            }
        }

        // 2. 打包后的应用：使用已解析的 resource_dir（含 backend 资源）
        if resource_dir.join("backend").exists() {
            println!("[PythonBackend] 使用 resource_dir: {:?}", resource_dir);
            let backend_dir = resource_dir.join("backend");
            // 检查是否有嵌入式 Python
            let embedded_python = resource_dir.join("python").join("bin").join("python3");
            println!("[PythonBackend] embedded_python exists? {}", embedded_python.exists());
            if embedded_python.exists() {
                // venv 已迁移到 app_data_dir（不写入 bundle 内部）
                let venv_python = self.resolve_venv_python(app, resource_dir);
                if let Some(vp) = venv_python {
                    println!("[PythonBackend] 找到已有 venv: {:?}", vp);
                    return Ok((vp, backend_dir));
                }
                println!("[PythonBackend] 返回 embedded_python 让 ensure_venv 创建 app_data_dir/venv");
                return Ok((embedded_python, backend_dir));
            }

            // 打包后应用不应使用系统 Python，但为了开发调试暂时回退
            if let Some(python) = self.find_system_python() {
                println!("[PythonBackend] 警告: 打包模式下使用系统 Python（缺少嵌入式 Python）");
                return Ok((python, backend_dir));
            }
        } else {
            println!("[PythonBackend] backend_dir {:?} 不存在", resource_dir.join("backend"));
        }

        Err(anyhow!("无法定位 Python 解释器和后端代码"))
    }

    /// 解析最终 resource_dir：优先 Tauri resource_dir；若其下无 backend
    /// （macOS 某些运行场景下 resource_dir 可能指向不可访问路径），
    /// 用 exe 路径硬推导 Contents/Resources（exe 一定在 Contents/MacOS/ 下，往上两层即 Contents/）
    fn resolve_resource_dir<R: Runtime>(&self, app: &AppHandle<R>) -> Result<PathBuf> {
        let mut resource_dir = app.path().resource_dir()
            .map_err(|e| anyhow!("无法获取 resource_dir: {}", e))?;

        if !resource_dir.join("backend").exists() {
            println!("[PythonBackend] resource_dir={:?} 下无 backend，尝试 exe 推导", resource_dir);
            if let Ok(exe_path) = std::env::current_exe() {
                if let Some(resources) = exe_path.parent()
                    .and_then(|p| p.parent())        // Contents
                    .map(|p| p.join("Resources"))    // Contents/Resources
                {
                    if resources.join("backend").exists() {
                        println!("[PythonBackend] exe 推导成功: {:?}", resources);
                        resource_dir = resources;
                    }
                }
            }
        }
        Ok(resource_dir)
    }

    /// 解析 venv 路径：优先 app_data_dir/venv，兼容旧 bundle 内 venv（首次升级自动迁移）
    fn resolve_venv_python<R: Runtime>(&self, app: &AppHandle<R>, resource_dir: &PathBuf) -> Option<PathBuf> {
        let app_data_dir = app.path().app_data_dir().ok();
        let target_venv = app_data_dir.as_ref().map(|d| d.join("venv"));

        // 1. 新位置：app_data_dir/venv
        if let Some(ref tv) = target_venv {
            let venv_python = tv.join("bin").join("python");
            if venv_python.exists() {
                return Some(venv_python);
            }

            // 2. 旧位置：bundle 内 resource_dir/venv → 迁移到 app_data_dir
            let old_venv = resource_dir.join("venv");
            let old_python = old_venv.join("bin").join("python");
            if old_python.exists() {
                println!("[PythonBackend] 发现旧 bundle 内 venv，迁移到 app_data_dir");
                if let Some(parent) = tv.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                match std::fs::rename(&old_venv, tv) {
                    Ok(_) => {
                        println!("[PythonBackend] venv 迁移成功");
                        return Some(venv_python);
                    }
                    Err(e) => {
                        eprintln!("[PythonBackend] venv 迁移失败: {}，将在新位置重新创建", e);
                        let _ = std::fs::remove_dir_all(&old_venv);
                    }
                }
            }
        }

        None
    }

    /// 查找系统 Python 解释器
    fn find_system_python(&self) -> Option<PathBuf> {
        let candidates = vec!["python3", "python"];
        for name in candidates {
            if let Ok(output) = std::process::Command::new("which")
                .arg(name)
                .output()
            {
                if output.status.success() {
                    let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    if !path.is_empty() {
                        return Some(PathBuf::from(path));
                    }
                }
            }
        }
        None
    }

    /// 确保 venv 存在（打包模式下首次启动自动创建）
    /// 使用嵌入式 Python 创建 venv 并安装依赖，用户无需安装 Python
    /// venv 位置：app_data_dir/venv（写入用户数据目录，不污染 bundle）
    fn ensure_venv<R: Runtime>(
        &self,
        app: &AppHandle<R>,
        resource_dir: &PathBuf,
        python_path: &PathBuf,
        backend_path: &PathBuf,
        progress_callback: Option<&Box<dyn Fn(&str, u8, &str) + Send>>,
    ) -> Result<PathBuf> {
        // 嵌入式 Python 判定使用已解析的 resource_dir（贯穿启动链，避免 sandbox 路径不一致）
        let embedded_python = resource_dir.join("python").join("bin").join("python3");
        if !embedded_python.exists() {
            // 没有嵌入式 Python（开发模式），直接返回
            return Ok(python_path.clone());
        }

        // 依赖完整性检查脚本：必须覆盖 mlx_audio.tts（TTS 核心）、
        // huggingface_hub>=1.5（mlx-audio/transformers 共同要求）、modelscope（ModelScope 下载源）。
        let check_code = r#"
import sys
errors = []
for mod in ['mlx_audio.tts', 'torch', 'fastapi', 'uvicorn', 'modelscope']:
    try:
        __import__(mod)
    except ImportError as e:
        errors.append(f'import {mod}: {e}')
try:
    import huggingface_hub
    hv = tuple(int(x) for x in huggingface_hub.__version__.split('.')[:3])
    if hv < (1, 5, 0):
        errors.append(f'huggingface_hub version {huggingface_hub.__version__} < 1.5.0')
except ImportError as e:
    errors.append(f'huggingface_hub: {e}')
if errors:
    print('DEPS_CHECK_FAILED:', '; '.join(errors), file=sys.stderr)
    sys.exit(1)
print('ok')
"#;

        // venv 放在 app_data_dir/venv（用户数据目录，bundle 只读）
        let app_data_dir = app.path().app_data_dir()
            .map_err(|e| anyhow!("无法获取 app_data_dir: {}", e))?;
        let _ = std::fs::create_dir_all(&app_data_dir);
        let venv_dir = app_data_dir.join("venv");
        let venv_python = venv_dir.join("bin").join("python");
        if venv_python.exists() {
            // 位置绑定校验（强制）：venv 的解释器 symlink 与 pyvenv.cfg 记录的是创建时的
            // 嵌入式 Python 绝对路径。应用移动位置（/Applications ↔ DMG 卷 ↔ bundle 目录）
            // 后 symlink 悬空或指向失效，必须在启动前重建，否则 uvicorn 无法运行。
            let cfg_path = venv_dir.join("pyvenv.cfg");
            let cfg = std::fs::read_to_string(&cfg_path).unwrap_or_default();
            let want_exec = embedded_python.to_string_lossy().to_string();
            let want_home = embedded_python.parent()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default();
            let bound_ok = cfg.lines().any(|l| {
                let t = l.trim();
                (t.starts_with("executable") && t.contains(&want_exec))
                    || (t.starts_with("home") && t.contains(&want_home))
            });
            if !bound_ok {
                println!("[PythonBackend] venv 绑定的 Python 路径与当前应用位置不一致（应用位置已变化），删除重建 venv");
                let _ = std::fs::remove_dir_all(&venv_dir);
            }
        }
        if venv_python.exists() {
            // 检查 venv 中关键依赖是否完整，任何一个失败都触发补装
            let check_output = std::process::Command::new(&venv_python)
                .arg("-c")
                .arg(check_code)
                .output();

            let deps_ok = match &check_output {
                Ok(out) => {
                    if !out.status.success() {
                        let stderr = String::from_utf8_lossy(&out.stderr);
                        println!("[PythonBackend] venv 依赖检查失败: {}", stderr.trim());
                    }
                    out.status.success()
                }
                Err(_) => false,
            };

            if deps_ok {
                println!("[PythonBackend] venv 已存在且依赖完整，跳过创建");
                if let Some(cb) = progress_callback {
                    cb("venv_ready", 90, "运行环境已就绪");
                }
                return Ok(venv_python);
            }

            // 依赖不完整：先尝试幂等补装（已装的包 pip 会跳过，避免删除重建重复下载 torch 等大依赖），
            // 补装后重新验证；仍失败才删除重建。补装前等待其他实例的安装进程结束（并发保护）。
            println!("[PythonBackend] venv 依赖不完整，尝试补装缺失依赖");
            let repair_files = vec![
                ("requirements.txt", 30, 60, "核心依赖"),
                ("requirements-mlx.txt", 60, 90, "MLX 加速依赖"),
            ];
            let mut repaired = true;
            for (filename, start_pct, end_pct, desc) in &repair_files {
                let req_file = backend_path.join(filename);
                if !req_file.exists() {
                    continue;
                }
                // 等待其他 voxNest 实例的 pip 安装结束（最多约 60s）
                for _ in 0..12 {
                    let pip_running = std::process::Command::new("pgrep")
                        .arg("-f")
                        .arg("pip install")
                        .output()
                        .map(|o| o.status.success())
                        .unwrap_or(false);
                    if !pip_running {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_secs(5));
                }
                if let Some(cb) = progress_callback {
                    cb("installing_deps", *start_pct, &format!("正在补装{}...", desc));
                }
                println!("[PythonBackend] 正在补装依赖: {:?}", req_file);
                let install_output = std::process::Command::new(&venv_python)
                    .arg("-m")
                    .arg("pip")
                    .arg("install")
                    .arg("-r")
                    .arg(&req_file)
                    .arg("--quiet")
                    .output();

                match &install_output {
                    Ok(out) if out.status.success() => {
                        println!("[PythonBackend] 依赖补装完成: {:?}", req_file);
                    }
                    _ => {
                        println!("[PythonBackend] 依赖补装失败: {:?}", req_file);
                        repaired = false;
                    }
                }
                if let Some(cb) = progress_callback {
                    cb("installing_deps", *end_pct, &format!("{}补装完成", desc));
                }
            }

            // 补装后再次验证依赖完整性
            let recheck_output = std::process::Command::new(&venv_python)
                .arg("-c")
                .arg(check_code)
                .output();
            let recheck_ok = match &recheck_output {
                Ok(out) => out.status.success(),
                Err(_) => false,
            };
            if repaired && recheck_ok {
                println!("[PythonBackend] venv 补装后依赖完整");
                if let Some(cb) = progress_callback {
                    cb("venv_ready", 90, "运行环境已就绪");
                }
                return Ok(venv_python);
            }

            // 仍失败：删除旧 venv 重建
            println!("[PythonBackend] 补装仍失败，删除旧 venv 并重建");
            let _ = std::fs::remove_dir_all(&venv_dir);
        }

        if let Some(cb) = progress_callback {
            cb("creating_venv", 15, "正在创建虚拟环境...");
        }

        println!("[PythonBackend] 首次启动：正在创建虚拟环境...");
        println!("[PythonBackend] 嵌入式 Python: {:?}", embedded_python);
        println!("[PythonBackend] venv 目标: {:?}", venv_dir);

        // 步骤1：创建 venv（--clear 清空旧内容干净重建，避免对残留 venv 原地升级失败）
        let create_output = std::process::Command::new(&embedded_python)
            .arg("-m")
            .arg("venv")
            .arg("--clear")
            .arg(&venv_dir)
            .output()
            .map_err(|e| anyhow!("创建 venv 失败: {}", e))?;

        if !create_output.status.success() {
            let stderr = String::from_utf8_lossy(&create_output.stderr);
            return Err(anyhow!("创建 venv 失败: {}", stderr));
        }
        println!("[PythonBackend] venv 创建成功");

        // 步骤2：升级 pip
        if let Some(cb) = progress_callback {
            cb("upgrading_pip", 25, "正在升级 pip...");
        }
        println!("[PythonBackend] 正在升级 pip...");
        let _ = std::process::Command::new(&venv_python)
            .arg("-m")
            .arg("pip")
            .arg("install")
            .arg("--upgrade")
            .arg("pip")
            .output();

        // 步骤3：安装依赖
        // 必须安装所有 requirements 文件：requirements.txt（核心依赖）+ requirements-mlx.txt（MLX加速）
        let requirements_files = vec![
            ("requirements.txt", 30, 60, "核心依赖"),
            ("requirements-mlx.txt", 60, 90, "MLX 加速依赖"),
        ];

        for (filename, start_pct, end_pct, desc) in &requirements_files {
            let req_file = backend_path.join(filename);
            if req_file.exists() {
                if let Some(cb) = progress_callback {
                    cb("installing_deps", *start_pct, &format!("正在安装{}（约需2-5分钟）...", desc));
                }
                println!("[PythonBackend] 正在安装依赖: {:?}", req_file);
                let install_output = std::process::Command::new(&venv_python)
                    .arg("-m")
                    .arg("pip")
                    .arg("install")
                    .arg("-r")
                    .arg(&req_file)
                    .arg("--quiet")
                    .output()
                    .map_err(|e| anyhow!("安装依赖失败: {}", e))?;

                if !install_output.status.success() {
                    let stderr = String::from_utf8_lossy(&install_output.stderr);
                    eprintln!("[PythonBackend] 依赖安装失败: {:?}\n{}", req_file, stderr);
                } else {
                    println!("[PythonBackend] 依赖安装完成: {:?}", req_file);
                }
                if let Some(cb) = progress_callback {
                    cb("installing_deps", *end_pct, &format!("{}安装完成", desc));
                }
            }
        }

        // 终验：venv 依赖不完整绝不允许启动后端（否则 uvicorn import 失败表现为"后端连不上"）
        let final_check = std::process::Command::new(&venv_python)
            .arg("-c")
            .arg(check_code)
            .output();
        if let Ok(out) = &final_check {
            if !out.status.success() {
                let stderr = String::from_utf8_lossy(&out.stderr);
                eprintln!("[PythonBackend] venv 依赖终验失败: {}", stderr.trim());
                return Err(anyhow!(
                    "依赖环境安装不完整，无法启动后端: {}",
                    stderr.trim()
                ));
            }
        }

        if let Some(cb) = progress_callback {
            cb("venv_ready", 90, "运行环境初始化完成");
        }
        println!("[PythonBackend] 虚拟环境初始化完成");
        Ok(venv_python)
    }
}

impl Drop for PythonBackend {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}
