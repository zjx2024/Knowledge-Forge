# 迁移与发布安全

## 迁移命令

以下只读命令不会创建迁移表或修改业务数据：

```bash
conda run -n knowledge-forge-py312 python scripts/manage_migrations.py plan
conda run -n knowledge-forge-py312 python scripts/manage_migrations.py status
```

显式执行全部版本化迁移：

```bash
conda run -n knowledge-forge-py312 python scripts/manage_migrations.py apply
```

服务启动仅执行标记为快速且安全的迁移。回填、并发索引等重操作必须由运维人员显式执行。迁移受 PostgreSQL advisory lock 保护，每一步独立记录 checksum、状态、耗时、执行模式和进度；失败后再次执行会从持久化状态继续。已应用迁移的 checksum 改变时必须停止并使用新的 forward-fix 迁移，禁止改写历史脚本。

## 备份恢复验证

正式迁移前先对合成临时库验证本机 `pg_dump` 和 `pg_restore` 链路：

```bash
conda run -n knowledge-forge-py312 python scripts/verify_postgres_backup_restore.py
```

若 PostgreSQL 客户端仅存在于容器中：

```bash
conda run -n knowledge-forge-py312 python scripts/verify_postgres_backup_restore.py \
  --docker-container knowledge-forge-postgres
```

脚本只创建名称以 `knowledge_forge_backup_` 开头的随机数据库，并写入绑定本次 run ID 的 `kf_backup_sentinel`。恢复后逐表比较数量和规范内容 hash；清理前再次核验 sentinel。它不会备份或修改业务数据库。

业务迁移前仍需按组织的备份制度生成真实备份，并在独立实例完成恢复演练。回滚优先通过新的 forward-fix 迁移完成，避免回写已应用迁移或破坏审计历史。

## 破坏性旧列清理

默认命令只展示计划：

```bash
conda run -n knowledge-forge-py312 python scripts/manage_migrations.py cleanup-legacy-columns
```

只有确认无数据库对象引用并持有可追溯备份后，才可同时提供备份标识与两个显式开关执行。服务启动永远不会执行该清理。

## 网络与凭据

裸机和 Docker published port 默认只绑定 `127.0.0.1`。开发环境绑定 `0.0.0.0` 必须显式设置危险开关；production/shared/UAT 环境还必须启用认证并提供运维批准。未启用兼容认证机制时，禁止部署到互联网、共享 UAT 或不可信网络。

PostgreSQL 及启用鉴权的 HugeGraph 不提供弱默认密码，缺少有效 secret 时服务拒绝启动。审计负载默认脱敏，保留期限与容量通过环境变量控制并由恢复作业定期清理。
