设计清晰。P1-04 是纯确定性文本清洗层：编码规范化 → HTML/样板剥离 → 空白/标点归一化 → 每步带可审计的变更日志。**第一步：CleaningRecord — 单步变更日志 DTO****第二步：各独立 Rule 实现****第三步：Cleaner — 组装规则流水线，产出 CleanedDocument****第四步：CleanerFactory — 从 AppConfig 构建业务类型对应的 Cleaner****第五步：ingestion 包 `__init__.py`****第六步：安装软依赖，全量验证**全部通过。更新 pyproject.toml 补上软依赖声明：P1-04 完成。核心思路说明如下：

---

## 核心思路与关键点

### 文件结构
```
core/ingestion/
  cleaning_record.py  — DTO：CleaningRecord（单步日志）、CleanedDocument（完整输出）
  rules.py            — 7 个独立纯函数规则
  cleaner.py          — Cleaner：按序组装规则，产出 CleanedDocument
  cleaner_factory.py  — CleanerFactory：按 business_type 构建带定制样板的 Cleaner
```

### `cleaning_record.py` 关键点

- `CleaningRecord` 是 frozen dataclass，每条记录不可变——审计日志不能被事后篡改。
- 每条记录包含 `chars_before / chars_after`，即使规则无变化也会写一条 `NOOP` 记录，保证日志完整性——合规审查要求能证明"这条规则运行了且无影响"，而不是"不知道有没有运行"。
- `CleanedDocument.total_chars_removed` 和 `.summary()` 给运维一行可读摘要，方便批量入库时的日志监控。

### `rules.py` 关键点

**规则顺序是有意设计的：**
1. `fix_encoding` 先修复乱码，后续规则面对的是合法 UTF-8
2. `strip_control_chars` 去除会干扰 HTML 解析的不可见字符
3. `normalize_unicode` NFC 标准化 + 全角转半角（从 PDF/Word 复制的中文文档大量含全角数字、标点）
4. `strip_html` 在 unicode 归一化之后，确保 HTML 实体能正确解码
5. `strip_boilerplate` 在空白归一化之前，保证正则行匹配准确
6. `normalize_whitespace` 在所有内容清理之后最后收尾
7. `fix_repeated_punct` 最后处理，防止前面步骤产生新的连续标点

**软依赖降级：**
- `ftfy` 不存在 → encoding_fix 写 NOOP 记录，流水线继续
- `bs4` 不存在 → HTML strip 退化为 regex 实现
- 两种降级都记录在 `CleaningRecord.detail` 里，不抛异常、不阻断入库

**`keyword` 字段不经过规则的原则：** 规则只处理 `text` 字段（`text` 类型）。`doc_id`、`chunk_id`、`business_type` 等 keyword 字段由上层保持原值传入，cleaner 完全不接触它们。

### `cleaner_factory.py` 关键点

- 正则 pattern 是代码不是 JSON 数据——在 Python 里有语法检查、可单测。
- 5 类业务各自有精准的样板模式：
  - **news**：社交分享按钮文字、阅读量行、来源/编辑署名
  - **policy**：签字栏、版本记录表行、受控文件戳
  - **workflow**：系统生成的节点 ID 行、状态行、纯时间戳行
  - **equipment**：零件清单脚注、QR 码行、保修免责声明
  - **quality_kb**：标准文档号、规范性引用文件章节头
- 未知 `business_type` 静默降级为通用 Cleaner，不抛异常——健壮性优先。