## ADDED Requirements

### Requirement: Checksum 文件下载

程序 SHALL 在下载 zip 文件后自动下载对应的 .CHECKSUM 文件。

#### Scenario: 下载 Checksum 文件

- **WHEN** zip 文件下载成功后
- **THEN** 自动下载对应的 .CHECKSUM 文件
- **AND** 保存到与 zip 文件相同的目录

---

### Requirement: 文件完整性校验

程序 SHALL 使用 SHA256 算法校验下载的文件完整性。

#### Scenario: 校验成功

- **WHEN** 下载完成后进行校验
- **AND** 计算的 SHA256 与 CHECKSUM 文件内容匹配
- **THEN** 打印 "校验通过: filename"

#### Scenario: 校验失败

- **WHEN** 计算的 SHA256 与 CHECKSUM 文件内容不匹配
- **THEN** 删除损坏的 zip 文件
- **AND** 打印 "校验失败，文件已删除: filename"
- **AND** 标记该文件下载失败

---

### Requirement: Checksum 格式处理

程序 SHALL 能正确解析 Binance 提供的 CHECKSUM 格式。

#### Scenario: 解析 Checksum

- **WHEN** 读取 .CHECKSUM 文件内容
- **THEN** 提取 SHA256 哈希值（忽略空白字符）
- **AND** 与本地文件计算值进行比对
