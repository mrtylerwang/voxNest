#!/usr/bin/env node
/**
 * voxNest prebuild — macOS (Apple Silicon) only
 * 在 Tauri build 之前准备好 src-tauri 下的资源目录：
 *   - frontend/app-live.html + app.css  →  src-tauri/dist/
 *   - backend/app/ + requirements*.txt →  src-tauri/backend/
 *   - bin/ffmpeg + bin/ffprobe          →  src-tauri/bin/
 *   - ★ 从 package.json 读版本号 → 写进 src-tauri/backend/app/main.py 的 APP_VERSION
 *
 * 版本号权威源 = package.json 的 "version" 字段。
 * prebuild 每次执行时自动同步 APP_VERSION，彻底消除源码 vs 打包副本的版本漂移。
 *
 * 嵌入式 Python 下载由 scripts/setup_python_runtime.sh 单独处理。
 */

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const TAURI = path.join(ROOT, 'src-tauri');

function log(msg) { console.log(`[prebuild] ${msg}`); }

function ensureDir(dir) { fs.mkdirSync(dir, { recursive: true }); }

function copyFile(src, dest) {
  if (!fs.existsSync(src)) {
    log(`  skip (not found): ${path.relative(ROOT, src)}`);
    return;
  }
  ensureDir(path.dirname(dest));
  fs.copyFileSync(src, dest);
  log(`  copy: ${path.relative(ROOT, src)} → ${path.relative(ROOT, dest)}`);
}

function copyDir(src, dest) {
  if (!fs.existsSync(src)) {
    log(`  skip (not found): ${path.relative(ROOT, src)}`);
    return;
  }
  ensureDir(path.dirname(dest));
  fs.rmSync(dest, { recursive: true, force: true });
  fs.cpSync(src, dest, { recursive: true });
  log(`  copy dir: ${path.relative(ROOT, src)} → ${path.relative(ROOT, dest)}`);
}

// ===== 读取权威版本号 =====
const PKG = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8'));
const VERSION = PKG.version;
log(`版本号: ${VERSION}（来自 package.json）`);

// 1. Frontend → src-tauri/dist
log('1/3 Frontend → src-tauri/dist');
ensureDir(path.join(TAURI, 'dist'));
copyFile(path.join(ROOT, 'frontend', 'app-live.html'), path.join(TAURI, 'dist', 'index.html'));
copyFile(path.join(ROOT, 'frontend', 'app.css'), path.join(TAURI, 'dist', 'app.css'));

// 2. Backend → src-tauri/backend（含版本号同步）
log('2/3 Backend → src-tauri/backend');
copyDir(path.join(ROOT, 'backend', 'app'), path.join(TAURI, 'backend', 'app'));
copyFile(path.join(ROOT, 'backend', 'requirements.txt'), path.join(TAURI, 'backend', 'requirements.txt'));
copyFile(path.join(ROOT, 'backend', 'requirements-mlx.txt'), path.join(TAURI, 'backend', 'requirements-mlx.txt'));

// ★ 关键：强制同步 APP_VERSION，消除源码 vs 打包副本漂移
const mainPyPath = path.join(TAURI, 'backend', 'app', 'main.py');
if (fs.existsSync(mainPyPath)) {
  let content = fs.readFileSync(mainPyPath, 'utf8');
  const before = content;
  content = content.replace(
    /^APP_VERSION\s*=\s*"[^"]*"/m,
    `APP_VERSION = "${VERSION}"`
  );
  if (content !== before) {
    fs.writeFileSync(mainPyPath, content);
    log(`  sync APP_VERSION → ${VERSION}`);
  } else {
    log(`  APP_VERSION 已是 ${VERSION}，跳过`);
  }
}

// 3. ffmpeg → src-tauri/bin（macOS 二进制，无 .exe）
log('3/3 ffmpeg → src-tauri/bin');
ensureDir(path.join(TAURI, 'bin'));
copyFile(path.join(ROOT, 'bin', 'ffmpeg'), path.join(TAURI, 'bin', 'ffmpeg'));
copyFile(path.join(ROOT, 'bin', 'ffprobe'), path.join(TAURI, 'bin', 'ffprobe'));
try { fs.chmodSync(path.join(TAURI, 'bin', 'ffmpeg'), 0o755); } catch (e) {}
try { fs.chmodSync(path.join(TAURI, 'bin', 'ffprobe'), 0o755); } catch (e) {}

log('Done.');
