# Security Policy

Please report suspected vulnerabilities privately to **zhouhaoyu@linkerbot.cn**.
Do not open a public GitHub issue for security reports. We aim to acknowledge
reports within 5 business days and to coordinate disclosure thereafter.

## Network listeners

The Viser viewer serves an unauthenticated HTTP/WebSocket server on a local
port (default 8080). Do not expose it to untrusted networks; tunnel over SSH
if remote access is required. Treat any deployment that binds the viewer to a
non-loopback address as a misconfiguration.
