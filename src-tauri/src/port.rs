//! 动态端口分配
//! 查找可用的本地端口

use std::net::TcpListener;

/// 查找一个可用的本地端口
pub fn find_free_port() -> Option<u16> {
    // 尝试绑定到端口 0，让操作系统分配一个可用端口
    match TcpListener::bind("127.0.0.1:0") {
        Ok(listener) => {
            match listener.local_addr() {
                Ok(addr) => Some(addr.port()),
                Err(_) => None,
            }
        }
        Err(_) => None,
    }
}

/// 检查指定端口是否可用
#[allow(dead_code)]
pub fn is_port_available(port: u16) -> bool {
    TcpListener::bind(("127.0.0.1", port)).is_ok()
}
