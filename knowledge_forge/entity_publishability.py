"""Deterministic publication eligibility for grounded graph entities.

The model extraction ledger is immutable source evidence.  This module builds
a cheap, versioned publication projection over that evidence: obvious parser
or document-metadata artefacts are rejected before embedding, uncertain but
potentially useful mentions remain auditable, and canonical entities are
published only from surviving support.  No rule depends on a project, domain
term, concrete document, entity id, or predicate name.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Mapping


ENTITY_PUBLISHABILITY_POLICY_VERSION = (
    "entity-publishability-structural-metadata-v13-terminal-fragment-gate"
)

ACCEPTED = "accepted"
CONDITIONAL = "conditional"
REJECTED = "rejected"

_ROLE_ALIASES = {
    "table-row": "table_row",
    "table row": "table_row",
    "row": "table_row",
    "table-cell": "table_cell",
    "table cell": "table_cell",
    "cell": "table_cell",
    "list-item": "list_item",
    "list item": "list_item",
    "image": "media_description",
    "figure": "media_description",
    "caption": "caption",
}
_STRUCTURALLY_PROTECTED_ROLES = frozenset({
    "list_item", "table_row", "table_cell",
})
_STRUCTURAL_FLAGS = frozenset({
    "coordinated_distinct_candidate",
    "section_label_candidate",
    "inline_label_candidate",
    "soft_wrap_surface_repaired",
})

# These are renderer-owned labels, not document/domain vocabulary.  A mention
# grounded only in one of these lines describes the retrieval envelope rather
# than the source document's subject matter.
_RENDERER_METADATA_LINE = re.compile(
    r"^\s*(?:"
    r"章节\s*ID|来源文档|分块|文档图片|图片来源|"
    r"section\s*id|source\s*document|document\s*source|chunk|"
    r"image\s*source"
    r")\s*[:：]",
    re.IGNORECASE,
)
_MEDIA_DESCRIPTION_LINE = re.compile(
    r"^\s*(?:图片标题|图片描述|图像描述|图表描述|图片文字|"
    r"图片中明确事实|图片中明确关系|"
    r"image\s*description|figure\s*description|visual\s*description)\s*[:：]",
    re.IGNORECASE,
)
_MEDIA_RENDERER_SURFACE = re.compile(
    r"^(?:图片标题|图片描述|图像描述|图表描述|图片文字|"
    r"图片中明确事实|图片中明确关系|"
    r"图片|图像|图表|图|照片|示意图|"
    r"image(?:\s+description)?|figure(?:\s+description)?|"
    r"visual\s+description|picture|photo|diagram|illustration|chart|plot)$",
    re.IGNORECASE,
)
_MEDIA_AXIS_RENDERER_SURFACE = re.compile(
    r"^(?:[xyz][ _-]?axis|horizontal[ _-]?axis|vertical[ _-]?axis|"
    r"横轴|横坐标轴|纵轴|纵坐标轴|坐标轴)$",
    re.IGNORECASE,
)
_MEDIA_MARKER_RENDERER_SURFACE = re.compile(
    r"^(?:(?:black|white|gr[ae]y|red|orange|yellow|green|blue|purple|"
    r"violet|pink|cyan|magenta|brown)[ _-]+"
    r"(?:circle|square|triangle|star|diamond|dot|line|curve|bar|stroke)"
    r"(?:[ _-]+markers?)?|"
    r"(?:黑|白|灰|红|橙|黄|绿|蓝|紫|粉|青|棕)色?"
    r"(?:圆|圆点|方块|正方形|三角形|星形|菱形|标记|线|曲线|柱|条))$",
    re.IGNORECASE,
)
_MEDIA_CHART_KIND_SURFACE = re.compile(
    r"^(?:(?:line|bar|column|area|scatter|bubble|pie|radar|box|boxplot|"
    r"histogram|heatmap|violin|waterfall|funnel|gauge|polar|network)\s+"
    r"(?:chart|plot|graph)|(?:折线|柱状|条形|面积|散点|气泡|饼|雷达|箱线|"
    r"直方|热力|小提琴|瀑布|漏斗|仪表|极坐标|网络)图)$",
    re.IGNORECASE,
)
_MEDIA_SERIES_RENDERER_SURFACE = re.compile(
    r"^(?:(?:data\s+)?series|trace|curve)\s+[A-Za-z0-9_.-]{1,32}$",
    re.IGNORECASE,
)
_FIGURE_CAPTION_SCAFFOLD = re.compile(
    r"(?:\b(?:figure|fig\.?|table)\s*[A-Za-z]?\d+(?:[.\-]\d+)*\b|"
    r"(?:^|\s)[图表]\s*[A-Za-z]?\d+(?:[.\-]\d+)*)"
    r".*\b(?:plot|chart)\s+of\b",
    re.IGNORECASE,
)
_FIGURE_CAPTION_SCAFFOLD_SURFACE = re.compile(
    r"^(?:plot|chart)$", re.IGNORECASE,
)
_SYSTEM_MEDIA_FIELD_SURFACE = re.compile(
    r"^(?:图片标题|图片描述|图像描述|图表描述|图片文字|"
    r"图片中明确事实|图片中明确关系|"
    r"image\s*description|figure\s*description|visual\s*description)$",
    re.IGNORECASE,
)
_BIBLIOGRAPHIC_IDENTIFIER_LINE = re.compile(
    r"^\s*(?:doi|e-?mail|email|电子邮箱|邮箱|网址|url|uri|"
    r"issn|isbn|cn\s*号|统一刊号)\s*[:：]",
    re.IGNORECASE,
)
_DOCUMENT_METADATA_FIELD_LINE = re.compile(
    r"^\s*[\[【]?\s*(?P<label>"
    r"关键词|关键字|中图分类号|文献标识码|基金项目|资助项目|"
    r"作者简介|作者单位|通信作者|通讯作者|邮编|邮政编码|"
    r"doi|e-?mail|email|电子邮箱|邮箱|网址|url|uri|"
    r"issn|isbn|cn\s*号|统一刊号|"
    r"keywords?|classification\s+(?:number|code)|funding|"
    r"corresponding\s+author|affiliation|address"
    r")\s*[\]】]?\s*[:：]",
    re.IGNORECASE,
)
_DOCUMENT_METADATA_FIELD_SURFACE = re.compile(
    r"^(?:"
    r"关键词|关键字|中图分类号|文献标识码|基金项目|资助项目|"
    r"作者简介|作者单位|通信作者|通讯作者|邮编|邮政编码|"
    r"doi|e-?mail|email|电子邮箱|邮箱|网址|url|uri|"
    r"issn|isbn|cn\s*号|统一刊号|"
    r"keywords?|classification\s+(?:number|code)|funding|"
    r"corresponding\s+author|affiliation|address"
    r")$",
    re.IGNORECASE,
)
_DOCUMENT_METADATA_FIELD_TOKEN = re.compile(
    r"(?:^|[\s,，;；|/（(\[【])\s*(?:"
    r"关键词|关键字|中图分类号|文献标识码|基金项目|资助项目|"
    r"作者简介|作者单位|通信作者|通讯作者|邮编|邮政编码|"
    r"doi|e-?mail|email|电子邮箱|邮箱|网址|url|uri|"
    r"issn|isbn|cn\s*号|统一刊号|"
    r"keywords?|classification\s+(?:number|code)|funding|"
    r"corresponding\s+author|affiliation|address"
    r")\s*(?:[\]】]\s*)?[:：]",
    re.IGNORECASE,
)
_BRACKETED_DOCUMENT_METADATA_FIELD = re.compile(
    r"[\[【]\s*(?:"
    r"关键词|关键字|中图分类号|文献标识码|基金项目|资助项目|"
    r"作者简介|作者单位|通信作者|通讯作者|邮编|邮政编码|"
    r"doi|e-?mail|email|电子邮箱|邮箱|网址|url|uri|"
    r"issn|isbn|cn\s*号|统一刊号|"
    r"keywords?|classification\s+(?:number|code)|funding|"
    r"corresponding\s+author|affiliation|address"
    r")\s*[\]】]",
    re.IGNORECASE,
)
_DOCUMENT_IDENTIFIER_CONTEXT = re.compile(
    r"(?:基金|资助|项目\s*(?:编号|号)|grant|fund(?:ing)?|"
    r"project\s*(?:id|number|no\.?|code)|"
    r"中图分类号|classification\s+(?:number|code)|"
    r"作者单位|affiliation|address|邮编|邮政编码|postal\s+code|"
    r"文献标识码|identifier|serial\s+(?:number|code))",
    re.IGNORECASE,
)
_CONTEXTUAL_SERIAL_IDENTIFIER = re.compile(
    r"^(?=.{5,}$)(?=.*\d)[A-Za-z0-9][A-Za-z0-9._/-]*$"
)
_PAGE_ONLY_HEADING = re.compile(
    r"^\s*#{0,6}\s*(?:第\s*\d+\s*页|page\s*\d+)\s*$",
    re.IGNORECASE,
)
_EMAIL = re.compile(
    r"^[^\s@<>]+@[^\s@<>]+\.[^\s@<>]+$", re.IGNORECASE,
)
_URL = re.compile(r"^(?:https?://|ftp://|www\.)\S+$", re.IGNORECASE)
_DOI = re.compile(
    r"^(?:doi\s*[:：]\s*)?(?:https?://(?:dx\.)?doi\.org/)?"
    r"10\.\d{4,9}/\S+$",
    re.IGNORECASE,
)
_SERIAL_IDENTIFIER = re.compile(
    r"^(?:(?:e-)?issn|isbn|cn)\s*[:：]?\s*[0-9A-Za-z./-]{6,}$",
    re.IGNORECASE,
)
_CODE_LIKE = re.compile(
    r"^(?=.*[A-Za-z0-9])(?:[A-Za-z_][A-Za-z0-9_.:/()\[\]-]*|"
    r"[A-Za-z0-9]+(?:[-_.:/][A-Za-z0-9]+)+)$"
)
_PRIVATE_OR_REPLACEMENT = re.compile(r"[\ufffd\ue000-\uf8ff]")

# Citation syntax is document structure, not domain vocabulary.  Requiring a
# numbered entry plus a publication-shaped signal avoids treating ordinary
# numbered requirements or procedures as bibliography entries.
_REFERENCE_ORDINAL = re.compile(
    r"^\s*(?:\[\s*\d{1,4}\s*\]|\d{1,4}\s*[.)、．])\s*"
)
_CITATION_MEDIUM = re.compile(
    r"\[\s*(?:J|C|M|D|R|S|P|N|EB\s*/\s*OL|DB\s*/\s*OL|CP\s*/\s*DK)"
    r"\s*\]",
    re.IGNORECASE,
)
_CITATION_YEAR = re.compile(r"(?<!\d)(?:18|19|20)\d{2}[a-z]?(?!\d)", re.IGNORECASE)
_CITATION_ISSUE_OR_PAGES = re.compile(
    r"(?:\b\d{1,4}\s*\(\s*\d{1,3}\s*\)\s*[:：,，]\s*\d{1,5}"
    r"|\b\d{1,4}\s*[:：]\s*\d{1,5}(?:\s*[-–—]\s*\d{1,5})?)"
)
_CITATION_LAYOUT_TOKEN = re.compile(
    r"^(?:\[\s*(?:\d{1,4}|J|C|M|D|R|S|P|N|EB\s*/\s*OL|"
    r"DB\s*/\s*OL|CP\s*/\s*DK)\s*\]|\d{1,4}\s*\(\s*\d{1,3}\s*\))$",
    re.IGNORECASE,
)
_CITATION_RANGE_FRAGMENT = re.compile(
    r"^(?:[\[【]\s*)?\d{1,4}\s*[-–—]\s*\d{1,4}"
    r"\s*(?:[\]】])$"
)
_INLINE_PUBLICATION_LOCATOR = re.compile(
    r"(?:\bdoi\s*[:：]\s*10\.\d{4,9}/\S+|"
    r"https?://(?:dx\.)?doi\.org/10\.\d{4,9}/\S+|https?://\S+)",
    re.IGNORECASE,
)
_REFERENCE_HEADING_LINE = re.compile(
    r"^\s*(?:参考文献|references?|bibliography)\s*$", re.IGNORECASE,
)
_FIGURE_TABLE_LOCATOR = re.compile(
    r"^(?:图|表|figure|fig\.?|table)\s*[A-Za-z]?\d+(?:[.\-]\d+)*$",
    re.IGNORECASE,
)
_EQUATION_LOCATOR = re.compile(
    r"^(?:(?:式|公式)\s*[（(]\s*(?:\d+(?:[.\-]\d+)*)?\s*[)）]|"
    r"(?:equation|eq\.?)\s*[（(]?\s*\d+(?:[.\-]\d+)*\s*[)）]?)$",
    re.IGNORECASE,
)
_SOFT_WRAP_HYPHEN = frozenset("-‐‑‒–—⁃")
_OCR_SPACED_ASCII_FORMULA_TOKEN = re.compile(
    r"^(?:[A-Za-z]\s+){2,}[A-Za-z]$"
)
_FORMULA_CONTEXT = re.compile(
    r"[=×÷∩∪∑∏√∥‖≤≥≈≠∈∉⊂⊆]"
)
_BARE_ASCII_FORMULA_INDEX = re.compile(r"^[a-z]$")

# Closed-class grammar tokens cannot become useful graph nodes merely because
# a model assigned a concrete entity type.  This list contains only language
# function words; it contains no project, predicate or domain vocabulary.
_FUNCTION_ONLY_SURFACES = frozenset({
    "的", "地", "得", "在", "于", "和", "与", "及", "或", "为", "是",
    "由", "被", "对", "向", "从", "至", "以", "将", "并", "而", "但",
})
_UNRESOLVED_DEICTIC_REFERENCE = re.compile(
    r"^(?:"
    r"前者|后者|双方|彼此|各自|其中|其他|其它|它们|他们|她们|"
    r"(?:上述|前述|以上|以下|这|那|该|此)?(?:两|二|三|四|五|几)者|"
    r"(?:上述|前述|以上|以下|这些|那些|该|此)(?:这些|那些)?"
    r"(?:方法|策略|机制|方案|"
    r"模型|系统|技术|内容|对象|结果|情况|工作|研究|文档|章节|部分|"
    r"问题|任务|过程|步骤|组件|模块|实体|关系)|"
    r"(?:各|两|多|若干)[\u3400-\u9fffA-Za-z0-9_]{1,16}(?:之)?间|"
    r"it|its|itself|they|them|their|theirs|themselves|"
    r"this|that|these|those|former|latter|both|either|neither|"
    r"each\s+other|one\s+another|respectively"
    r")$",
    re.IGNORECASE,
)
_CHINESE_AUXILIARY_PREDICATE = re.compile(
    r"(?:能够|可以|需要|必须|应当|应该|将会|正在|已经|有望|"
    r"无需|无法|不能|不会|能否|(?:也|又|仍|还)能)"
)
_CHINESE_BARE_NECESSITY = re.compile(r"需(?!求)")
_CHINESE_WILL_CONSTRUCTION = re.compile(r"将(?:从|对|把|由|在|向|以|通过)")
_CHINESE_LIMITED_MEANS_CLAUSE = re.compile(
    r"^(?:仅|只)(?:在|通过|凭借|依靠|当).{2,}(?:才|来|以|时)"
)
_CHINESE_CONTINUATION_CLAUSE = re.compile(
    r"^(?:并(?!行|发|列|集|购|网|口|联)|且|也|又|仍|还|从而|进而)"
)
_CHINESE_ASPECT_CLAUSE = re.compile(
    r"^(?:[\u3400-\u9fff]{2,5}了[\u3400-\u9fff]{4,}|"
    r"[\u3400-\u9fff]{2,12}中[\u3400-\u9fff]{1,6}着"
    r"[\u3400-\u9fff]{2,})$"
)
_CHINESE_FINITE_RELATIONAL_FRAME = re.compile(
    r"(?:(?:通常|一般|主要|大致|往往)?分(?:为|成)|"
    r"受[\u3400-\u9fff]{1,4}于)"
)
_CHINESE_EXPLICIT_CLAIM_FRAME = re.compile(
    r"(?:通过.{2,}(?:来|以|实现|建模|强化|保证|提升|降低|减少)|"
    r"(?:实现|完成|强化|缓解|提升|降低|减少|避免|导致|引发)了|"
    r"可避免.{2,}导致|以适应|"
    r"从而.{2,}(?:实现|提升|降低|保证))"
)
_CHINESE_NOMINAL_HEAD = re.compile(
    r"(?:方法|策略|机制|技术|系统|模型|算法|流程|方案|框架|标准|"
    r"协议|规则|要求|功能|服务|接口|项目|工程|平台|体系|能力|"
    r"任务|活动|事件|过程|类型|类别)$"
)
_CHINESE_QUANTIFIED_SUBJECT = re.compile(
    r"^(?:各|每个?|所有|任何)[\u3400-\u9fffA-Za-z0-9_]{1,16}"
    r"(?:在|将|对|向|由|被)"
)
_ENGLISH_FINITE_OR_MODAL = re.compile(
    r"\b(?:is|are|was|were|has|have|had|can|could|may|might|must|"
    r"shall|should|will|would|does|did)\b",
    re.IGNORECASE,
)
_RELATION_PROMOTABLE_CONDITIONAL_REASONS = frozenset({
    "short_generic_surface",
    "media_entity_requires_relation_support",
})

# Presentation/layout vocabulary is intentionally limited to visual form.  It
# is evaluated only inside a parser-rendered media-description line, so terms
# such as "line" in ordinary engineering prose are unaffected.
_VISUAL_LAYOUT_NOUN = re.compile(
    r"(?:\b(?:background|border|margin|page|whitespace|white\s+space|"
    r"horizontal\s+line|vertical\s+line|square\s+pattern|layout)\b|"
    r"背景|边框|頁面|页面|空白页|留白|横线|橫線|竖线|豎線|版式|布局)",
    re.IGNORECASE,
)
_VISUAL_LAYOUT_QUALIFIER = re.compile(
    r"(?:\b(?:blank|empty|white|black|horizontal|vertical|single|plain|"
    r"near\s+the\s+(?:top|bottom)|arranged)\b|"
    r"空白|空的|白色|黑色|水平|垂直|单一|單一|顶部|頂部|底部|排列)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MentionPublishabilityDecision:
    decision: str
    reason: str
    score: float
    protected: bool = False
    features: tuple[str, ...] = ()
    policy_version: str = ENTITY_PUBLISHABILITY_POLICY_VERSION


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _normalized_role(value: Any) -> str:
    role = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _ROLE_ALIASES.get(role, role)


def _source_role_for_span(
    metadata: Mapping[str, Any], raw: Mapping[str, Any], start: int, end: int,
) -> str:
    for item in metadata.get("structural_spans") or ():
        if not isinstance(item, Mapping):
            continue
        try:
            span_start = int(item.get("start"))
            span_end = int(item.get("end"))
        except (TypeError, ValueError):
            continue
        if span_start <= start < end <= span_end:
            role = _normalized_role(item.get("source_role"))
            if role:
                return role
    return _normalized_role(
        metadata.get("structural_role")
        or metadata.get("source_role")
        or metadata.get("chunk_role")
        or metadata.get("cell_role")
        or raw.get("structural_role")
    )


def _line_for_span(content: str, start: int, end: int) -> str:
    if not content or not (0 <= start < end <= len(content)):
        return ""
    left = content.rfind("\n", 0, start) + 1
    right = content.find("\n", end)
    if right < 0:
        right = len(content)
    return content[left:right].strip()


def _identifier_kind(surface: str) -> str:
    compact = surface.strip().rstrip(".,;，。；")
    if _EMAIL.fullmatch(compact):
        return "email"
    if _DOI.fullmatch(compact):
        return "publication_identifier"
    if _URL.fullmatch(compact):
        return "external_locator"
    if _SERIAL_IDENTIFIER.fullmatch(compact):
        return "publication_identifier"
    return ""


def _identifier_only_line(value: str) -> bool:
    parts = [
        item.strip().strip("()[]{}（）【】<>")
        for item in re.split(r"[,，;；]", str(value or "").strip())
        if item.strip()
    ]
    return bool(parts) and all(_identifier_kind(item) for item in parts)


def _is_visual_layout_phrase(surface: str) -> bool:
    return bool(
        _VISUAL_LAYOUT_NOUN.search(surface)
        and _VISUAL_LAYOUT_QUALIFIER.search(surface)
    )


def _renderer_generated_media_context(
    metadata: Mapping[str, Any], role: str, line: str,
) -> bool:
    """Return whether a mention lives in our generated visual envelope.

    Axis placeholders and colour/shape marker names can be legitimate text in
    an ordinary document.  They are renderer syntax only when the chunk or
    structural span is explicitly media-derived, so this gate does not leak
    visual conventions into general entity policy.
    """

    block_types = {
        str(value or "").strip().lower()
        for value in metadata.get("block_types") or ()
    }
    return bool(
        role in {"media_description", "caption"}
        or "image" in block_types
        or "figure" in block_types
        or _MEDIA_DESCRIPTION_LINE.match(line)
    )


def bibliography_entry_context(value: str) -> bool:
    """Recognize high-confidence citation-entry text without domain terms."""

    line = unicodedata.normalize("NFKC", str(value or "")).strip()
    if not line:
        return False
    ordinal = bool(_REFERENCE_ORDINAL.match(line))
    medium = bool(_CITATION_MEDIUM.search(line))
    locator = bool(_INLINE_PUBLICATION_LOCATOR.search(line))
    publication_shape = bool(
        _CITATION_YEAR.search(line) and _CITATION_ISSUE_OR_PAGES.search(line)
    )
    return bool(
        (ordinal and (medium or locator or publication_shape))
        or (medium and (_CITATION_YEAR.search(line) or ordinal))
    )


@lru_cache(maxsize=128)
def bibliography_reference_spans(value: str) -> tuple[tuple[int, int], ...]:
    """Return high-confidence, possibly wrapped bibliography regions."""

    content = str(value or "")
    if not content:
        return ()
    records: list[tuple[int, int, str]] = []
    offset = 0
    for raw in content.splitlines(keepends=True):
        end = offset + len(raw)
        records.append((offset, end, raw.rstrip("\r\n")))
        offset = end
    if offset < len(content) or not records:
        records.append((offset, len(content), content[offset:]))

    explicit = [
        index for index, (_, _, line) in enumerate(records)
        if _REFERENCE_HEADING_LINE.fullmatch(
            unicodedata.normalize("NFKC", line).strip()
        )
    ]
    if explicit:
        return ((records[explicit[0]][0], len(content)),)

    starts = [
        index for index, (_, _, line) in enumerate(records)
        if _REFERENCE_ORDINAL.match(unicodedata.normalize("NFKC", line))
    ]
    ranges: list[tuple[int, int]] = []
    validated_line_ranges: list[tuple[int, int]] = []
    for position, line_index in enumerate(starts):
        next_index = (
            starts[position + 1] if position + 1 < len(starts)
            else min(len(records), line_index + 16)
        )
        if next_index <= line_index:
            continue
        candidate = "\n".join(
            records[index][2] for index in range(line_index, next_index)
        )
        if not bibliography_entry_context(candidate):
            continue
        validated_line_ranges.append((line_index, next_index))

    validated_entry_count = len(validated_line_ranges)
    if validated_line_ranges:
        first_valid_start = validated_line_ranges[0][0]
        # Recover only a short wrapped citation tail immediately before the
        # first numbered entry. This handles page boundaries without treating
        # a whole preceding article or numbered procedure as references.
        prefix_start = max(0, first_valid_start - 10)
        prefix = [records[index][2] for index in range(
            prefix_start, first_valid_start,
        )]
        for local_index in range(len(prefix) - 1, -1, -1):
            if not _CITATION_MEDIUM.search(
                unicodedata.normalize("NFKC", prefix[local_index])
            ):
                continue
            tail = unicodedata.normalize(
                "NFKC", "\n".join(prefix[local_index:])
            )
            if _CITATION_YEAR.search(tail) or _CITATION_ISSUE_OR_PAGES.search(tail):
                validated_line_ranges.insert(0, (
                    prefix_start + max(0, local_index - 1),
                    first_valid_start,
                ))
            break

    if validated_entry_count >= 2:
        # Once a contiguous reference run is independently established, keep
        # a final wrapped numbered entry even when its journal/year tail is on
        # the next page. Never extend to markers preceding the first verified
        # citation, which could be ordinary numbered requirements.
        first_valid_start = validated_line_ranges[0][0]
        covered_starts = {
            line_index
            for line_start, line_end in validated_line_ranges
            for line_index in starts
            if line_start <= line_index < line_end
        }
        for position, line_index in enumerate(starts):
            if line_index < first_valid_start or line_index in covered_starts:
                continue
            next_index = (
                starts[position + 1] if position + 1 < len(starts)
                else min(len(records), line_index + 16)
            )
            validated_line_ranges.append((line_index, next_index))

    for line_start, line_end in validated_line_ranges:
        ranges.append((
            records[line_start][0],
            records[min(len(records), line_end) - 1][1],
        ))
    if not ranges:
        return ()
    ranges.sort()
    merged: list[list[int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return tuple((start, end) for start, end in merged)


def bibliography_span_context(
    content: str, start: int, end: int,
) -> bool:
    return any(
        region_start <= start < end <= region_end
        for region_start, region_end in bibliography_reference_spans(content)
    )


def canonical_display_surface(value: Any) -> str:
    """Remove parser soft-wrap spacing from a canonical display name only."""

    normalized = unicodedata.normalize("NFKC", str(value or ""))
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(
        r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])", "", normalized,
    )
    normalized = re.sub(r"\s+([，。！？；：、）】》〉])", r"\1", normalized)
    normalized = re.sub(r"([（【《〈])\s+", r"\1", normalized)
    return normalized


def _document_metadata_label(surface: str, line: str) -> bool:
    candidate = unicodedata.normalize("NFKC", str(surface or "")).strip(" ：:")
    if not _DOCUMENT_METADATA_FIELD_SURFACE.fullmatch(candidate):
        return False
    normalized_line = unicodedata.normalize("NFKC", str(line or ""))
    return bool(
        _DOCUMENT_METADATA_FIELD_TOKEN.search(normalized_line)
        or _BRACKETED_DOCUMENT_METADATA_FIELD.search(normalized_line)
    )


def _document_metadata_identifier(surface: str, line: str) -> bool:
    """Recognize identifier-shaped metadata values only in field context."""

    candidate = unicodedata.normalize("NFKC", str(surface or "")).strip()
    normalized_line = unicodedata.normalize("NFKC", str(line or ""))
    return bool(
        _CONTEXTUAL_SERIAL_IDENTIFIER.fullmatch(candidate)
        and _DOCUMENT_IDENTIFIER_CONTEXT.search(normalized_line)
    )


def _looks_like_clause_surface(
    surface: str,
    entity_type: str,
    raw: Mapping[str, Any],
    role: str,
    *,
    content: str,
    start: int,
    end: int,
) -> bool:
    """Detect strong finite-clause shapes without using domain vocabulary.

    Entity type is deliberately not trusted as an escape hatch.  A model can
    label a complete assertion as an Event, Organization or Technology while
    still selecting the wrong canonical boundary.  Explicit proposition
    arguments remain eligible because they are intentional event identities;
    noun-like list/table items remain unaffected because this gate requires a
    finite/modal, connective or aspect construction rather than merely a long
    surface.
    """

    del entity_type, role
    normalized = re.sub(r"\s+", " ", surface).strip()
    cjk = "".join(re.findall(r"[\u3400-\u9fff]", normalized))
    words = re.findall(r"[A-Za-z]+", normalized)
    proposition_argument = raw.get("proposition_argument") is True
    # Compact event propositions can be useful identities. Long finite claims
    # belong in relation/evidence records instead of becoming giant graph
    # nodes, even when a model tagged the entire clause as an Event argument.
    if proposition_argument and (
        (cjk and len(cjk) <= 12)
        or (not cjk and len(words) <= 8)
    ):
        return False
    if len(cjk) >= 7:
        if (
            _CHINESE_AUXILIARY_PREDICATE.search(cjk)
            or _CHINESE_WILL_CONSTRUCTION.search(cjk)
            or _CHINESE_LIMITED_MEANS_CLAUSE.search(cjk)
            or _CHINESE_QUANTIFIED_SUBJECT.match(cjk)
            or _CHINESE_ASPECT_CLAUSE.fullmatch(cjk)
            or (
                len(cjk) >= 16
                and _CHINESE_EXPLICIT_CLAIM_FRAME.search(cjk)
                and not _CHINESE_NOMINAL_HEAD.search(cjk)
            )
            or (
                _CHINESE_FINITE_RELATIONAL_FRAME.search(cjk)
                and not _CHINESE_NOMINAL_HEAD.search(cjk)
            )
        ):
            return True
        for match in _CHINESE_BARE_NECESSITY.finditer(cjk):
            before = cjk[:match.start()]
            after = cjk[match.end():]
            if (
                len(before) >= 2
                and len(after) >= 2
                and not before.endswith(("按", "所", "刚", "供", "必"))
            ):
                return True
        previous = content[:start].rstrip()[-1:] if start > 0 else ""
        if (
            previous in {",", "，", ";", "；", "。", "!", "！", "?", "？"}
            and _CHINESE_CONTINUATION_CLAUSE.match(cjk)
        ):
            return True
        # A typed model mention can stop at one side of a coordinated
        # subordinate clause, especially across a PDF soft wrap.  The source
        # bytes immediately following the span prove that the selected text is
        # not a complete identity ("X 在 ... 和 ... 的同时"), while ordinary
        # location-bearing noun phrases have no such right-hand continuation.
        right_cjk = "".join(re.findall(
            r"[\u3400-\u9fff]", content[end:min(len(content), end + 64)],
        ))
        if (
            "在" in cjk[2:]
            and len(cjk.rsplit("在", 1)[-1]) >= 3
            and right_cjk.startswith(("和", "与", "及"))
            and re.search(r"(?:的同时|过程中|条件下|情况下)", right_cjk[:32])
        ):
            return True
    if len(words) >= 5 and _ENGLISH_FINITE_OR_MODAL.search(normalized):
        return True
    return False


def classify_mention_publishability(
    *,
    surface: str,
    entity_type: str,
    content: str,
    start: int,
    end: int,
    chunk_metadata: Mapping[str, Any] | None = None,
    raw: Mapping[str, Any] | None = None,
) -> MentionPublishabilityDecision:
    """Classify one exact mention without a model or domain-specific rules."""

    metadata = _mapping(chunk_metadata)
    raw_value = _mapping(raw)
    stripped = str(surface or "").strip()
    normalized = unicodedata.normalize("NFKC", stripped)
    role = _source_role_for_span(metadata, raw_value, start, end)
    line = _line_for_span(content, start, end)
    features: list[str] = []

    identifier_kind = _identifier_kind(normalized)
    if identifier_kind:
        features.append(identifier_kind)
    if role:
        features.append(f"source_role:{role}")

    if _PRIVATE_OR_REPLACEMENT.search(normalized):
        return MentionPublishabilityDecision(
            REJECTED, "ocr_private_or_replacement_fragment", 0.0,
            features=tuple(features + ["invalid_unicode_glyph"]),
        )

    if _SYSTEM_MEDIA_FIELD_SURFACE.fullmatch(normalized.strip(" ：:")):
        return MentionPublishabilityDecision(
            REJECTED, "media_renderer_label", 0.0,
            features=tuple(features + ["renderer_owned"]),
        )

    if _FIGURE_TABLE_LOCATOR.fullmatch(normalized):
        return MentionPublishabilityDecision(
            REJECTED, "document_figure_table_locator", 0.0,
            features=tuple(features + ["source_locator"]),
        )

    if _EQUATION_LOCATOR.fullmatch(normalized):
        return MentionPublishabilityDecision(
            REJECTED, "document_equation_locator", 0.0,
            features=tuple(features + ["source_locator"]),
        )

    # A mention ending at a discretionary hyphen is only the left half of a
    # word when either the right half starts on the following layout line or
    # the parser clipped the source at a chunk/block boundary.  The latter is
    # common for PDF extraction but is format-independent: DOCX/Markdown
    # converters can also terminate a block at the same character.  Keep the
    # exact source bytes in the extraction audit, but never turn an unfinished
    # alphanumeric token into a retrievable identity.
    if (
        normalized
        and normalized[-1] in _SOFT_WRAP_HYPHEN
        and re.search(r"[\w\u3400-\u9fff]", normalized[:-1])
        and (
            re.match(
                r"^[ \t]*\r?\n[ \t]*[\w\u3400-\u9fff]", content[end:]
            )
            or not content[end:].strip()
        )
    ):
        return MentionPublishabilityDecision(
            REJECTED, "soft_wrap_truncated_surface", 0.0,
            features=tuple(features + ["incomplete_source_span"]),
        )

    if (
        _OCR_SPACED_ASCII_FORMULA_TOKEN.fullmatch(normalized)
        and _FORMULA_CONTEXT.search(line)
    ):
        return MentionPublishabilityDecision(
            REJECTED, "ocr_spaced_formula_operator", 0.0,
            features=tuple(features + ["formula_syntax"]),
        )

    # Keep formulas in searchable source evidence, but do not publish a whole
    # operator expression or a bare Latin summation/index variable as an
    # entity merely because it participated in a model-proposed relation.
    # Qualified symbols such as ``θ_j`` remain eligible: unlike a lone ``j``,
    # they can carry a document-defined identity useful to retrieval.
    if _FORMULA_CONTEXT.search(normalized):
        return MentionPublishabilityDecision(
            REJECTED, "formula_expression_surface", 0.0,
            features=tuple(features + ["formula_syntax"]),
        )
    if (
        _BARE_ASCII_FORMULA_INDEX.fullmatch(normalized)
        and _FORMULA_CONTEXT.search(line)
    ):
        return MentionPublishabilityDecision(
            REJECTED, "formula_index_symbol", 0.0,
            features=tuple(features + ["formula_syntax"]),
        )

    # Renderer metadata must never become graph semantics merely because an
    # endpoint-recall candidate happened to cover the same exact bytes.
    if _RENDERER_METADATA_LINE.match(line):
        return MentionPublishabilityDecision(
            REJECTED, "renderer_metadata_line", 0.0,
            features=tuple(features + ["renderer_owned"]),
        )

    if _PAGE_ONLY_HEADING.fullmatch(line):
        return MentionPublishabilityDecision(
            REJECTED, "renderer_page_heading", 0.0,
            features=tuple(features + ["renderer_owned"]),
        )

    if _BIBLIOGRAPHIC_IDENTIFIER_LINE.match(line):
        return MentionPublishabilityDecision(
            REJECTED, "document_metadata_identifier", 0.0,
            features=tuple(features + ["bibliographic_field"]),
        )

    if (
        bibliography_entry_context(line)
        or bibliography_span_context(content, start, end)
    ):
        return MentionPublishabilityDecision(
            REJECTED, "bibliography_entry_context", 0.0,
            features=tuple(features + ["retrieval_only_citation_entry"]),
        )

    if _document_metadata_label(normalized, line):
        return MentionPublishabilityDecision(
            REJECTED, "document_metadata_field_label", 0.0,
            features=tuple(features + ["document_metadata_label"]),
        )

    if _document_metadata_identifier(normalized, line):
        return MentionPublishabilityDecision(
            REJECTED, "document_metadata_identifier", 0.0,
            features=tuple(features + ["document_property"]),
        )

    if identifier_kind and _identifier_only_line(line):
        return MentionPublishabilityDecision(
            REJECTED, "standalone_external_identifier", 0.0,
            features=tuple(features + ["standalone_identifier_line"]),
        )

    if (
        _MEDIA_DESCRIPTION_LINE.match(line)
        and _MEDIA_RENDERER_SURFACE.fullmatch(normalized)
    ):
        return MentionPublishabilityDecision(
            REJECTED, "media_renderer_label", 0.0,
            features=tuple(features + ["renderer_owned"]),
        )

    media_context = _renderer_generated_media_context(metadata, role, line)
    if media_context and (
        _MEDIA_RENDERER_SURFACE.fullmatch(normalized)
        or _MEDIA_AXIS_RENDERER_SURFACE.fullmatch(normalized)
        or _MEDIA_MARKER_RENDERER_SURFACE.fullmatch(normalized)
        or _MEDIA_CHART_KIND_SURFACE.fullmatch(normalized)
        or _MEDIA_SERIES_RENDERER_SURFACE.fullmatch(normalized)
    ):
        return MentionPublishabilityDecision(
            REJECTED, "media_visual_encoding_token", 0.0,
            features=tuple(features + ["renderer_owned", "visual_encoding"]),
        )

    if (
        _FIGURE_CAPTION_SCAFFOLD_SURFACE.fullmatch(normalized)
        and _FIGURE_CAPTION_SCAFFOLD.search(line)
    ):
        return MentionPublishabilityDecision(
            REJECTED, "figure_caption_scaffold", 0.0,
            features=tuple(features + ["source_locator", "caption_scaffold"]),
        )

    if (
        _MEDIA_DESCRIPTION_LINE.match(line)
        and str(entity_type or "GenericEntity") == "GenericEntity"
        and _is_visual_layout_phrase(normalized)
    ):
        return MentionPublishabilityDecision(
            REJECTED, "media_layout_description", 0.0,
            features=tuple(features + ["visual_layout_only"]),
        )

    # External locators belong to source provenance/document properties.  They
    # remain in the searchable chunk and extraction audit, but never become
    # graph entities or relation endpoints.
    if identifier_kind:
        return MentionPublishabilityDecision(
            REJECTED, "external_identifier_is_provenance", 0.0,
            features=tuple(dict.fromkeys(features + ["document_property"])),
        )

    if (
        _CITATION_LAYOUT_TOKEN.fullmatch(normalized)
        or _CITATION_RANGE_FRAGMENT.fullmatch(normalized)
    ):
        return MentionPublishabilityDecision(
            REJECTED, "citation_layout_token", 0.0,
            features=tuple(features + ["bibliography_layout"]),
        )

    if normalized.casefold() in _FUNCTION_ONLY_SURFACES:
        return MentionPublishabilityDecision(
            REJECTED, "non_referential_function_word", 0.0,
            features=tuple(features + ["grammar_token"]),
        )

    if _UNRESOLVED_DEICTIC_REFERENCE.fullmatch(normalized):
        return MentionPublishabilityDecision(
            REJECTED, "unresolved_deictic_reference", 0.0,
            features=tuple(features + ["grammar_reference"]),
        )

    lexical = re.sub(r"[^\w\u3400-\u9fff]+", "", normalized, flags=re.UNICODE)
    if len(lexical) <= 1 and not _CODE_LIKE.fullmatch(normalized):
        structurally_supported = bool(
            role in _STRUCTURALLY_PROTECTED_ROLES
            or raw_value.get("endpoint_candidate") is True
            or any(raw_value.get(flag) is True for flag in _STRUCTURAL_FLAGS)
        )
        short_features = list(features) + ["low_information_surface"]
        if structurally_supported:
            short_features.append("relation_endpoint_candidate")
        return MentionPublishabilityDecision(
            CONDITIONAL, "short_generic_surface", 0.5,
            protected=structurally_supported,
            features=tuple(dict.fromkeys(short_features)),
        )

    if _looks_like_clause_surface(
        normalized,
        entity_type,
        raw_value,
        role,
        content=content,
        start=start,
        end=end,
    ):
        return MentionPublishabilityDecision(
            CONDITIONAL, "clause_like_surface", 0.35,
            protected=False,
            features=tuple(features + ["sentence_shape"]),
        )

    protected = bool(
        role in _STRUCTURALLY_PROTECTED_ROLES
        or raw_value.get("endpoint_candidate") is True
        or any(raw_value.get(flag) is True for flag in _STRUCTURAL_FLAGS)
        or (_CODE_LIKE.fullmatch(normalized) and not identifier_kind)
    )
    if protected:
        protected_features = list(features)
        if role in _STRUCTURALLY_PROTECTED_ROLES:
            protected_features.append("eligible_source_structure")
        if raw_value.get("endpoint_candidate") is True:
            protected_features.append("relation_endpoint_candidate")
        if _CODE_LIKE.fullmatch(normalized) and not identifier_kind:
            protected_features.append("bounded_identifier")
        return MentionPublishabilityDecision(
            ACCEPTED, "source_or_structure_supported", 1.0,
            protected=True,
            features=tuple(dict.fromkeys(protected_features)),
        )

    if role in {"media_description", "caption"}:
        return MentionPublishabilityDecision(
            CONDITIONAL, "media_entity_requires_relation_support", 0.6,
            features=tuple(features + ["media_source"]),
        )

    return MentionPublishabilityDecision(
        ACCEPTED, "source_backed_entity", 0.9,
        protected=False,
        features=tuple(features),
    )


def canonical_relation_candidate_eligible(canonical: Mapping[str, Any]) -> bool:
    """Treat missing metadata as eligible for backward-compatible snapshots."""

    metadata = _mapping(canonical.get("metadata"))
    publishability = _mapping(metadata.get("entity_publishability"))
    value = publishability.get("relation_candidate_eligible")
    return True if value is None else bool(value)


def canonical_retrieval_eligible(canonical: Mapping[str, Any]) -> bool:
    metadata = _mapping(canonical.get("metadata"))
    publishability = _mapping(metadata.get("entity_publishability"))
    value = publishability.get("retrieval_eligible")
    return True if value is None else bool(value)


def canonical_graph_publishable(canonical: Mapping[str, Any]) -> bool:
    metadata = _mapping(canonical.get("metadata"))
    publishability = _mapping(metadata.get("entity_publishability"))
    value = publishability.get("graph_publishable")
    return True if value is None else bool(value)


def aggregate_canonical_publishability(
    *,
    canonicals: Iterable[Mapping[str, Any]],
    assignments: Iterable[Mapping[str, Any]],
    mentions: Iterable[Mapping[str, Any]],
    relation_endpoints: Iterable[Mapping[str, Any]] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build an O(M+C+R) canonical publication audit projection."""

    canonical_values = [dict(item) for item in canonicals]
    mention_by_key = {
        str(item.get("mention_key") or item.get("mention_id") or ""): dict(item)
        for item in mentions
        if str(item.get("mention_key") or item.get("mention_id") or "")
    }
    mention_keys_by_canonical: dict[str, list[str]] = defaultdict(list)
    for assignment in assignments:
        canonical_key = str(assignment.get("canonical_key") or "")
        mention_key = str(assignment.get("mention_key") or "")
        if canonical_key and mention_key in mention_by_key:
            mention_keys_by_canonical[canonical_key].append(mention_key)

    relation_backed = {
        str(value)
        for endpoint in relation_endpoints
        for value in (
            endpoint.get("source_canonical_key"),
            endpoint.get("target_canonical_key"),
        )
        if str(value or "")
    }
    rows: list[dict[str, Any]] = []
    decisions: Counter[str] = Counter()
    for canonical in canonical_values:
        canonical_key = str(canonical.get("canonical_key") or "")
        support = [
            mention_by_key[key]
            for key in mention_keys_by_canonical.get(canonical_key, ())
        ]
        accepted = [
            item for item in support
            if str(item.get("filter_decision") or ACCEPTED) == ACCEPTED
        ]
        conditional = [
            item for item in support
            if str(item.get("filter_decision") or "") == CONDITIONAL
        ]
        promotable_conditional = [
            item for item in conditional
            if str(item.get("filter_reason") or "")
            in _RELATION_PROMOTABLE_CONDITIONAL_REASONS
        ]
        documents = {int(item.get("document_id") or 0) for item in support}
        chunks = {int(item.get("chunk_id") or 0) for item in support}
        protected_count = sum(
            bool(_mapping(item.get("provenance")).get("publishability_protected"))
            for item in support
        )
        promotable_protected_count = sum(
            bool(_mapping(item.get("provenance")).get("publishability_protected"))
            for item in promotable_conditional
        )
        repeated_conditional = bool(
            promotable_conditional
            and (
                len({
                    int(item.get("document_id") or 0)
                    for item in promotable_conditional
                }) > 1
                or len({
                    int(item.get("chunk_id") or 0)
                    for item in promotable_conditional
                }) > 1
            )
        )
        has_relation = canonical_key in relation_backed
        resolver_quarantined = bool(
            _mapping(canonical.get("metadata")).get("quarantined")
        )

        if resolver_quarantined:
            decision = "resolver_quarantined"
            reason = "identity_resolution_quarantined"
            relation_eligible = False
            retrieval_eligible = False
            graph_publishable = False
        elif accepted:
            decision = "accepted"
            reason = (
                "accepted_mention_with_verified_relation"
                if has_relation else "accepted_for_entity_retrieval"
            )
            relation_eligible = True
            retrieval_eligible = True
            graph_publishable = has_relation
        elif has_relation and promotable_conditional:
            decision = "relation_backed"
            reason = "verified_relation_support"
            relation_eligible = True
            retrieval_eligible = True
            graph_publishable = True
        elif promotable_conditional and (
            promotable_protected_count or repeated_conditional
        ):
            decision = "conditional"
            reason = (
                "protected_conditional_support"
                if promotable_protected_count else "repeated_conditional_support"
            )
            relation_eligible = True
            retrieval_eligible = repeated_conditional
            graph_publishable = False
        else:
            decision = "withheld"
            reason = "insufficient_graph_support"
            relation_eligible = False
            retrieval_eligible = False
            graph_publishable = False

        decisions[decision] += 1
        rows.append({
            "canonical_key": canonical_key,
            "audit": {
                "policy_version": ENTITY_PUBLISHABILITY_POLICY_VERSION,
                "decision": decision,
                "reason": reason,
                "accepted_mentions": len(accepted),
                "conditional_mentions": len(conditional),
                "promotable_conditional_mentions": len(
                    promotable_conditional
                ),
                "supporting_documents": len(documents),
                "supporting_chunks": len(chunks),
                "protected_mentions": int(protected_count),
                "verified_relation_backed": has_relation,
                "relation_candidate_eligible": relation_eligible,
                "retrieval_eligible": retrieval_eligible,
                "graph_publishable": graph_publishable,
            },
        })

    metrics = {
        "policy_version": ENTITY_PUBLISHABILITY_POLICY_VERSION,
        "canonicals": len(canonical_values),
        "decisions": dict(sorted(decisions.items())),
        "relation_candidate_eligible": sum(
            bool(item["audit"]["relation_candidate_eligible"]) for item in rows
        ),
        "retrieval_eligible": sum(
            bool(item["audit"]["retrieval_eligible"]) for item in rows
        ),
        "graph_publishable": sum(
            bool(item["audit"]["graph_publishable"]) for item in rows
        ),
        "relation_backed_promotions": sum(
            item["audit"]["decision"] == "relation_backed" for item in rows
        ),
    }
    return rows, metrics


def project_publishable_snapshot(
    snapshot: Mapping[str, Any],
    *,
    mentions: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the clean graph projection while retaining the persisted ledger."""

    result = dict(snapshot)
    canonicals = [
        dict(item) for item in snapshot.get("canonicals") or ()
        if canonical_graph_publishable(item)
    ]
    canonical_keys = {
        str(item.get("canonical_key") or "") for item in canonicals
    }
    assignments = [
        dict(item) for item in snapshot.get("assignments") or ()
        if str(item.get("canonical_key") or "") in canonical_keys
    ]
    assignment_mentions = {
        str(item.get("mention_key") or "") for item in assignments
    }
    endpoints = [dict(item) for item in snapshot.get("relation_endpoints") or ()]
    dangling = [
        str(item.get("relation_key") or "")
        for item in endpoints
        if str(item.get("source_canonical_key") or "") not in canonical_keys
        or str(item.get("target_canonical_key") or "") not in canonical_keys
    ]
    if dangling:
        raise ValueError(
            "publishable entity projection contains dangling relation endpoints: "
            + ",".join(dangling[:10])
        )
    result["canonicals"] = canonicals
    result["assignments"] = assignments
    result["relation_endpoints"] = endpoints
    if mentions is not None:
        result["mentions"] = [
            dict(item) for item in mentions
            if str(item.get("mention_key") or item.get("mention_id") or "")
            in assignment_mentions
        ]
    return result


__all__ = [
    "ACCEPTED",
    "CONDITIONAL",
    "REJECTED",
    "ENTITY_PUBLISHABILITY_POLICY_VERSION",
    "MentionPublishabilityDecision",
    "aggregate_canonical_publishability",
    "bibliography_entry_context",
    "bibliography_reference_spans",
    "bibliography_span_context",
    "canonical_display_surface",
    "canonical_graph_publishable",
    "canonical_relation_candidate_eligible",
    "canonical_retrieval_eligible",
    "classify_mention_publishability",
    "project_publishable_snapshot",
]
