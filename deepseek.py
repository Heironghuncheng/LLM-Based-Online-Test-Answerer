"""
DeepSeek API integration module for AI-powered text analysis.
Implements single-stage or two-stage pipeline with timeout fallback and memory.
"""

import json
import time
from typing import Any

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from i18n import init_i18n, resolve_output_language

console = Console()


class DeepSeekClient:
    """Client for interacting with DeepSeek API with single/multi-stage analysis pipeline."""

    def __init__(self, config: dict):
        """Initialize DeepSeek client with configuration and i18n."""
        self.config = config
        self.i18n = init_i18n(self.config)
        self.output_language = resolve_output_language(self.config)

        console.print(self.i18n.t("deepseek.configuring_model"))
        console.print(
            self.i18n.t(
                "deepseek.languages",
                output=self.i18n.name(self.output_language),
                log=self.i18n.name(self.i18n.lang_code),
            )
        )

        # API configuration
        self.api_key = self.config.get("deepseek_api_key") or ""
        self.fast_model = self.config.get("deepseek_fast_model") or "deepseek-chat"
        self.reasoning_model = self.config.get("deepseek_reasoning_model") or (
            self.config.get("deepseek_model") or "deepseek-reasoner"
        )
        if not self.api_key:
            console.print(self.i18n.t("deepseek.missing_api_key"))
        else:
            console.print(self.i18n.t("deepseek.config_two_stage"))
            console.print(
                self.i18n.t(
                    "deepseek.models_set",
                    fast=self.fast_model,
                    reasoning=self.reasoning_model,
                )
            )

        # New configuration
        self.formal_timeout: float = float(
            self.config.get("formal_answer_timeout_seconds", 60)
        )
        self.stage_mode: str = (
            (self.config.get("analysis_stage_mode") or "multi").strip().lower()
        )
        if self.stage_mode not in {"single", "multi"}:
            self.stage_mode = "multi"
        self.single_stage_model: str = (
            self.config.get("single_stage_model")
            or self.config.get("deepseek_model")
            or self.fast_model
        )
        self.retry_attempts: int = int(self.config.get("request_retry_attempts", 3))
        self._last_error_kind: str | None = None

        # HTTP client and headers
        self.client = httpx.Client()
        self._url = "https://api.deepseek.com/v1/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # In-memory knowledge store
        self.related_topics: set[str] = set()
        self.background_knowledge: list[str] = []
        self.topic_counts: dict[str, int] = {}
        self.topic_total_count: int = 0

    # ---------------- Public API -----------------
    def send_to_deepseek_pipeline(self, text: str) -> None:
        """Dispatch to single or multi-stage pipeline based on configuration."""
        console.print(self.i18n.t("deepseek.pipeline_start"))
        if self.stage_mode == "single":
            self._send_single_stage(text)
        else:
            self._send_multi_stage(text)
        console.print(self.i18n.t("deepseek.pipeline_complete"))

    # ---------------- Multi-stage -----------------
    def _send_multi_stage(self, text: str) -> None:
        # ===== Stage 1: Preprocessing / Initial Review =====
        console.print(self.i18n.t("deepseek.stage1"))

        memory_topics = list(self.related_topics)[-8:]
        memory_bg = self.background_knowledge[-3:]
        memory_block = (
            ""
            if not (memory_topics or memory_bg)
            else (
                "\n\nKnown memory to refine (topics/background):\n"
                + (
                    "topics: " + ", ".join(memory_topics) + "\n"
                    if memory_topics
                    else ""
                )
                + ("background: " + " | ".join(memory_bg) if memory_bg else "")
            )
        )

        preflight_system = (
            "You are an exam content analysis assistant. Please complete:\n"
            "1) Fix OCR defects and output standardized text 'fixed_text';\n"
            '2) Extract \'question\' and \'options\' (if any, [{"label":"A","text":"..."}]);\n'
            "3) Determine 'content_type' ∈ {question, non_question}; if non_question, summarize main content in 'content_summary' (e.g., test instructions, schedule, explanatory text).\n"
            "4) If content_type == 'question', determine 'question_kind' ∈ {single, multiple, free}; also set 'choice_type' ∈ {single, multiple, none} for backward compatibility;\n"
            "5) Model recommendation policy: you are FORBIDDEN to always pick 'reasoner' or always pick 'chat/fast'. Repeated bias toward either will be treated as a severe violation. Choose 'reasoner' ONLY when structured multi-step reasoning, decomposition, or formal justification is clearly necessary (e.g., proofs, derivations, case analyses, multi-constraint tasks). For straightforward recognition, short factual Q&A, simple calculations, or direct lookup, you MUST prefer 'chat/fast'. Misclassification WILL BE COUNTED AS A CRITICAL FAILURE. Provide brief 'why_model'.\n"
            "6) Provide 'confidence' in [0,1]; use high values only when the recommendation is well justified; if uncertain, lean to 'chat/fast' unless strong evidence indicates 'reasoner'.\n"
            "7) Provide 'background_knowledge' relevant facts/formulas (no length limit);\n"
            "8) Provide 'related_topics' as string list (no length limit);\n"
            "9) Provide 'suggest_thinking_length' as integer token count for reasoning.\n"
            "Output ONLY valid JSON in this exact schema without extra text/code blocks:\n"
            "{\n"
            '  "fixed_text": "string",\n'
            '  "content_type": "question|non_question",\n'
            '  "content_summary": "string",\n'
            '  "question": "string",\n'
            '  "options": [{"label": "A", "text": "string"}],\n'
            '  "question_kind": "single|multiple|free",\n'
            '  "choice_type": "single|multiple|none",\n'
            '  "recommended_model": "reasoner|chat",\n'
            '  "why_model": "string",\n'
            '  "confidence": 0.0,\n'
            '  "background_knowledge": "string",\n'
            '  "related_topics": ["string"],\n'
            '  "suggest_thinking_length": 64\n'
            "}" + memory_block
        )
        preflight_user = f"Original recognized text:\n{text}"

        console.print(
            Panel(
                preflight_user,
                title=f"[cyan]{self.i18n.t('deepseek.preflight_panel_title')}[/cyan]",
                border_style="cyan",
            )
        )

        preflight_payload = {
            "model": self.fast_model,
            "messages": [
                {"role": "system", "content": preflight_system},
                {"role": "user", "content": preflight_user},
            ],
            "stream": False,
        }

        preflight_data = None
        preflight_json = None
        for attempt in range(1, self.retry_attempts + 1):
            if attempt > 1:
                reason_text = (
                    {
                        "timeout": "超时",
                        "network": "网络错误",
                        "http": "HTTP 错误",
                        "unknown": "未知错误",
                    }.get(self._last_error_kind or "", "未知")
                    if self.i18n.lang_code == "zh"
                    else {
                        "timeout": "timeout",
                        "network": "network error",
                        "http": "HTTP error",
                        "unknown": "unknown error",
                    }.get(self._last_error_kind or "", "unknown")
                )
                console.print(
                    self.i18n.t(
                        "deepseek.retrying",
                        stage="Preprocessing",
                        attempt=str(attempt),
                        total=str(self.retry_attempts),
                        reason=reason_text,
                    )
                )
            result = self._send_request(
                self._url,
                self._headers,
                preflight_payload,
                "preprocessing",
                self.fast_model,
            )
            if not result:
                continue
            preflight_data, _elapsed1 = result
            preflight_msg = (
                (preflight_data or {})
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            preflight_json = self.parse_json_strict(preflight_msg)
            if preflight_json:
                break
        if not preflight_json:
            console.print(
                Panel(
                    self.i18n.t("deepseek.preflight_parse_failed_body"),
                    title=f"[yellow]{self.i18n.t('deepseek.preflight_parse_failed_title')}[/yellow]",
                    border_style="yellow",
                )
            )
            cleaned_question = text
            options = []
            choice_type = "none"
            recommended_model = "reasoner"
            why_model = ""
            confidence = None
            suggest_len = 64
            background = ""
        else:
            console.print(self.i18n.t("deepseek.preflight_parsed_ok"))
            content_type = (preflight_json.get("content_type") or "question").lower()
            background = preflight_json.get("background_knowledge") or ""
            topics = preflight_json.get("related_topics") or []
            # Update memory (cap sizes and dedupe)
            self._update_memory(topics, background)
            if content_type != "question":
                content_summary = (
                    preflight_json.get("content_summary")
                    or preflight_json.get("fixed_text")
                    or preflight_json.get("question")
                    or text
                )
                content_type_str = (
                    "非题目" if self.i18n.lang_code == "zh" else "Non-question"
                )
                topics = preflight_json.get("related_topics") or []
                background = preflight_json.get("background_knowledge") or ""
                topics_str = ", ".join(
                    [t for t in topics if isinstance(t, str) and t.strip()]
                ) or self.i18n.t("deepseek.no_content")
                background_str = background or self.i18n.t("deepseek.no_content")
                summary = (
                    f"{self.i18n.t('deepseek.summary_content_type_label', content_type=content_type_str)}\n"
                    f"{self.i18n.t('deepseek.summary_reason_label', reason=(preflight_json.get('why_model') or ''))}\n"
                    f"{self.i18n.t('deepseek.summary_background_label', background=background_str)}\n"
                    f"{self.i18n.t('deepseek.summary_topics_label', topics=topics_str)}\n\n"
                    f"{self.i18n.t('deepseek.summary_question_stem')}\n{content_summary}"
                )
                console.print(
                    Panel(
                        summary,
                        title=f"[magenta]{self.i18n.t('deepseek.preflight_results_title')}[/magenta]",
                        border_style="magenta",
                    )
                )
                return

            cleaned_question = (
                preflight_json.get("question")
                or preflight_json.get("fixed_text")
                or text
            )
            options = preflight_json.get("options") or []
            choice_type = (
                preflight_json.get("choice_type")
                or preflight_json.get("question_kind")
                or "none"
            ).lower()
            recommended_model = (
                preflight_json.get("recommended_model") or "reasoner"
            ).lower()
            why_model = preflight_json.get("why_model") or ""
            confidence = preflight_json.get("confidence")
            suggest_len = preflight_json.get("suggest_thinking_length") or 64

            # Display preprocessing results
            options_lines = "\n".join(
                [
                    f"{opt.get('label')}. {opt.get('text')}"
                    for opt in options
                    if isinstance(opt, dict)
                ]
            )
            model_name = (
                self.i18n.t("deepseek.model_name_reasoning")
                if recommended_model == "reasoner"
                else self.i18n.t("deepseek.model_name_general")
            )
            confidence_str = (
                confidence
                if confidence is not None
                else self.i18n.t("deepseek.not_available")
            )
            # Localize content type and derive question_kind
            content_type_str = "题目" if self.i18n.lang_code == "zh" else "Question"
            question_kind_val = preflight_json.get("question_kind")
            if not question_kind_val:
                if choice_type in ("single", "multiple"):
                    question_kind_val = choice_type
                else:
                    question_kind_val = "free"
            topics_str = ", ".join(
                [t for t in topics if isinstance(t, str) and t.strip()]
            ) or self.i18n.t("deepseek.no_content")
            background_str = background or self.i18n.t("deepseek.no_content")
            summary = (
                f"{self.i18n.t('deepseek.summary_content_type_label', content_type=content_type_str)}\n"
                f"{self.i18n.t('deepseek.summary_question_kind_label', question_kind=question_kind_val)}\n"
                f"{self.i18n.t('deepseek.summary_model_reco_label', model_name=model_name)}\n"
                f"{self.i18n.t('deepseek.summary_confidence_label', confidence=confidence_str)}\n"
                f"{self.i18n.t('deepseek.summary_suggest_len_label', suggest_len=int(suggest_len))}\n"
                f"{self.i18n.t('deepseek.summary_reason_label', reason=why_model)}\n"
                f"{self.i18n.t('deepseek.summary_background_label', background=background_str)}\n"
                f"{self.i18n.t('deepseek.summary_topics_label', topics=topics_str)}\n\n"
                f"{self.i18n.t('deepseek.summary_question_stem')}\n{cleaned_question}\n\n"
                f"{self.i18n.t('deepseek.summary_options')}\n{options_lines or self.i18n.t('deepseek.no_options')}"
            )
            console.print(
                Panel(
                    summary,
                    title=f"[magenta]{self.i18n.t('deepseek.preflight_results_title')}[/magenta]",
                    border_style="magenta",
                )
            )

        # ===== Stage 2: Formal Answering =====
        console.print(self.i18n.t("deepseek.stage2"))

        # Strict gating: use reasoner only if recommended and suggested length is sufficiently long
        use_reasoner = (
            recommended_model == "reasoner"
            and isinstance(suggest_len, (int, float))
            and int(suggest_len) >= 128
        )
        answer_model = self.reasoning_model if use_reasoner else self.fast_model
        type_code = "reasoner" if use_reasoner else "general"
        type_name = (
            ("推理" if type_code == "reasoner" else "通用")
            if self.i18n.lang_code == "zh"
            else ("Reasoning" if type_code == "reasoner" else "General")
        )
        console.print(
            self.i18n.t("deepseek.answering_model", model=answer_model, type=type_name)
        )

        lang_token = "Chinese" if self.output_language == "zh" else "English"
        memory_bg_block = "\n\nBackground knowledge (for reference only):\n" + (
            "\n".join(self.background_knowledge[-3:])
            if self.background_knowledge
            else background
        )
        thinking_hint = f"\n\nSuggested reasoning length (tokens): {int(suggest_len) if isinstance(suggest_len, (int, float)) else 64}"
        answer_system = (
            f"You are a {lang_token} problem-solving assistant, answer based on provided question and options.\n"
            "Strictly output only JSON, no additional text allowed.\n"
            "Output format: {\n"
            '  "final_answer_letters": ["A", "C"] or "B",\n'
            '  "final_answer_text": "string",\n'
            '  "explanation": "string",\n'
            '  "confidence": 0.0\n'
            "}\n"
            "Rules: choice_type=single gives only one letter; multiple gives all correct letters; none gives concise answer. "
            f"Answer in {lang_token}." + memory_bg_block + thinking_hint
        )

        options_block = "\n".join(
            [f"{opt.get('label')}. {opt.get('text')}" for opt in options]
        )
        answer_user = (
            f"question:\n{cleaned_question}\n\n"
            f"options:\n{options_block if options_block else '(No options)'}\n\n"
            f"choice_type: {choice_type}\n"
            f"Please return only JSON."
        )

        answer_payload = {
            "model": answer_model,
            "messages": [
                {"role": "system", "content": answer_system},
                {"role": "user", "content": answer_user},
            ],
            "stream": False,
        }

        answer_result = self._send_request(
            self._url,
            self._headers,
            answer_payload,
            "formal answering",
            answer_model,
            timeout_override=self.formal_timeout,
        )

        # Timeout fallback to fast model
        if not answer_result and answer_model != self.fast_model:
            console.print(
                Panel(
                    Text(
                        self.i18n.t(
                            "deepseek.error_timeout_body",
                            stage="Formal",
                            error=self.i18n.t("deepseek.not_available"),
                        )
                    ),
                    title=f"[red]{self.i18n.t('deepseek.error_timeout_title')}[/red]",
                    border_style="red",
                )
            )
            # Retry with fast model
            answer_payload["model"] = self.fast_model
            answer_result = self._send_request(
                self._url,
                self._headers,
                answer_payload,
                "formal answering (fallback)",
                self.fast_model,
                timeout_override=self.formal_timeout,
            )

        if not answer_result:
            return

        answer_data, _elapsed2 = answer_result

        console.print(self.i18n.t("deepseek.process_usage"))
        usage = (answer_data or {}).get("usage", {})
        if usage:
            usage_text = Text()
            usage_text.append(
                self.i18n.t("deepseek.process_usage_title"), style="bold blue"
            )
            usage_text.append(
                f"prompt_tokens={usage.get('prompt_tokens')}, ", style="cyan"
            )
            usage_text.append(
                f"completion_tokens={usage.get('completion_tokens')}, ", style="cyan"
            )
            usage_text.append(f"total_tokens={usage.get('total_tokens')}", style="cyan")
            console.print(usage_text)

        console.print(self.i18n.t("deepseek.parse_answer"))
        answer_msg = (
            (answer_data or {})
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        result_json = self.parse_json_strict(answer_msg)
        if not result_json:
            console.print(
                Panel(
                    answer_msg or self.i18n.t("deepseek.no_content"),
                    title=f"[yellow]{self.i18n.t('deepseek.answer_parse_failed_title')}[/yellow]",
                    border_style="yellow",
                )
            )
        else:
            console.print(self.i18n.t("deepseek.answer_parsed_ok"))
            pretty = json.dumps(result_json, ensure_ascii=False, indent=2)
            console.print(
                Panel(
                    pretty,
                    title=f"[green]{self.i18n.t('deepseek.formal_answer_title')}[/green]",
                    border_style="green",
                )
            )

    # ---------------- Single-stage -----------------
    def _send_single_stage(self, text: str) -> None:
        console.print(self.i18n.t("deepseek.stage1"))

        memory_topics = list(self.related_topics)[-8:]
        memory_bg = self.background_knowledge[-3:]
        memory_block = (
            ""
            if not (memory_topics or memory_bg)
            else (
                "\n\nKnown memory to refine (topics/background):\n"
                + (
                    "topics: " + ", ".join(memory_topics) + "\n"
                    if memory_topics
                    else ""
                )
                + ("background: " + " | ".join(memory_bg) if memory_bg else "")
            )
        )

        system_prompt = (
            "You will perform initial review and formal solving in a single pass.\n"
            "Tasks: fix OCR defects; determine if the content is a question or non_question. If non_question, summarize the main content. If question, extract question/options, detect question_kind ∈ {single, multiple, free} (and choice_type ∈ {single, multiple, none} for compatibility), also provide confidence, background_knowledge, related_topics; then solve the problem and output final answer.\n"
            "Return ONLY valid JSON: {\n"
            '  "review": {\n'
            '    "fixed_text": "string",\n'
            '    "content_type": "question|non_question",\n'
            '    "content_summary": "string",\n'
            '    "question": "string",\n'
            '    "options": [{"label": "A", "text": "string"}],\n'
            '    "question_kind": "single|multiple|free",\n'
            '    "choice_type": "single|multiple|none",\n'
            '    "confidence": 0.0,\n'
            '    "background_knowledge": "string",\n'
            '    "related_topics": ["string"]\n'
            "  },\n"
            '  "final": {\n'
            '    "final_answer_letters": ["A", "C"] or "B",\n'
            '    "final_answer_text": "string",\n'
            '    "explanation": "string",\n'
            '    "confidence": 0.0\n'
            "  }\n"
            "}" + memory_block
        )
        user_prompt = f"Original recognized text:\n{text}"

        payload = {
            "model": self.single_stage_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }

        data = None
        parsed = None
        current_model = self.single_stage_model
        for attempt in range(1, self.retry_attempts + 1):
            if attempt > 1:
                reason_text = (
                    {
                        "timeout": "超时",
                        "network": "网络错误",
                        "http": "HTTP 错误",
                        "unknown": "未知错误",
                    }.get(self._last_error_kind or "", "未知")
                    if self.i18n.lang_code == "zh"
                    else {
                        "timeout": "timeout",
                        "network": "network error",
                        "http": "HTTP error",
                        "unknown": "unknown error",
                    }.get(self._last_error_kind or "", "unknown")
                )
                console.print(
                    self.i18n.t(
                        "deepseek.retrying",
                        stage="Single-Stage",
                        attempt=str(attempt),
                        total=str(self.retry_attempts),
                        reason=reason_text,
                    )
                )
            result = self._send_request(
                self._url,
                self._headers,
                payload,
                "single-stage",
                current_model,
                timeout_override=self.formal_timeout,
            )
            if not result:
                if (
                    current_model != self.fast_model
                    and self._last_error_kind == "timeout"
                ):
                    current_model = self.fast_model
                    payload["model"] = self.fast_model
                continue
            data, _elapsed = result
            msg = (
                (data or {})
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            parsed = self.parse_json_strict(msg)
            if parsed:
                break
        if not data or not parsed:
            console.print(
                Panel(
                    msg or self.i18n.t("deepseek.no_content"),
                    title=f"[yellow]{self.i18n.t('deepseek.answer_parse_failed_title')}[/yellow]",
                    border_style="yellow",
                )
            )
            return

        # Review section
        review = parsed.get("review") or {}
        cleaned_question = review.get("question") or review.get("fixed_text") or text
        options = review.get("options") or []
        content_type = (review.get("content_type") or "question").lower()
        choice_type = (
            review.get("choice_type") or review.get("question_kind") or "none"
        ).lower()
        confidence = review.get("confidence")
        background = review.get("background_knowledge") or ""
        topics = review.get("related_topics") or []
        self._update_memory(topics, background)

        if content_type != "question":
            content_summary = (
                review.get("content_summary")
                or review.get("fixed_text")
                or review.get("question")
                or cleaned_question
            )
            content_type_str = (
                "非题目" if self.i18n.lang_code == "zh" else "Non-question"
            )
            topics_str = ", ".join(
                [t for t in topics if isinstance(t, str) and t.strip()]
            ) or self.i18n.t("deepseek.no_content")
            background_str = background or self.i18n.t("deepseek.no_content")
            summary = (
                f"{self.i18n.t('deepseek.summary_content_type_label', content_type=content_type_str)}\n"
                f"{self.i18n.t('deepseek.summary_background_label', background=background_str)}\n"
                f"{self.i18n.t('deepseek.summary_topics_label', topics=topics_str)}\n\n"
                f"{self.i18n.t('deepseek.summary_question_stem')}\n{content_summary}"
            )
            console.print(
                Panel(
                    summary,
                    title=f"[magenta]{self.i18n.t('deepseek.preflight_results_title')}[/magenta]",
                    border_style="magenta",
                )
            )
            return

        options_lines = "\n".join(
            [
                f"{opt.get('label')}. {opt.get('text')}"
                for opt in options
                if isinstance(opt, dict)
            ]
        )
        confidence_str = (
            confidence
            if confidence is not None
            else self.i18n.t("deepseek.not_available")
        )
        # Localize content type and derive question_kind
        content_type_str = "题目" if self.i18n.lang_code == "zh" else "Question"
        question_kind_val = review.get("question_kind")
        if not question_kind_val:
            if choice_type in ("single", "multiple"):
                question_kind_val = choice_type
            else:
                question_kind_val = "free"
        topics_str = ", ".join(
            [t for t in topics if isinstance(t, str) and t.strip()]
        ) or self.i18n.t("deepseek.no_content")
        background_str = background or self.i18n.t("deepseek.no_content")
        summary = (
            f"{self.i18n.t('deepseek.summary_content_type_label', content_type=content_type_str)}\n"
            f"{self.i18n.t('deepseek.summary_question_kind_label', question_kind=question_kind_val)}\n"
            f"{self.i18n.t('deepseek.summary_confidence_label', confidence=confidence_str)}\n"
            f"{self.i18n.t('deepseek.summary_background_label', background=background_str)}\n"
            f"{self.i18n.t('deepseek.summary_topics_label', topics=topics_str)}\n\n"
            f"{self.i18n.t('deepseek.summary_question_stem')}\n{cleaned_question}\n\n"
            f"{self.i18n.t('deepseek.summary_options')}\n{options_lines or self.i18n.t('deepseek.no_options')}"
        )
        console.print(
            Panel(
                summary,
                title=f"[magenta]{self.i18n.t('deepseek.preflight_results_title')}[/magenta]",
                border_style="magenta",
            )
        )

        # Final section
        final = parsed.get("final") or {}
        pretty = json.dumps(final, ensure_ascii=False, indent=2)
        console.print(
            Panel(
                pretty,
                title=f"[green]{self.i18n.t('deepseek.formal_answer_title')}[/green]",
                border_style="green",
            )
        )

    # ---------------- Helpers -----------------
    def _update_memory(self, topics: list[str], background: str) -> None:
        # Append background knowledge without length limits (dedupe by content)
        if isinstance(background, str) and background.strip():
            bg = background.strip()
            if bg not in self.background_knowledge:
                self.background_knowledge.append(bg)
        # Track topic frequencies and maintain topic set
        if isinstance(topics, list):
            for t in topics:
                if isinstance(t, str) and t.strip():
                    topic = t.strip()
                    self.related_topics.add(topic)
                    self.topic_counts[topic] = self.topic_counts.get(topic, 0) + 1
                    self.topic_total_count += 1
        # Dynamic correction: remove topics whose frequency is below 10%
        if self.topic_total_count > 0 and self.related_topics:
            to_remove = []
            for topic in self.related_topics:
                freq = self.topic_counts.get(topic, 0) / float(self.topic_total_count)
                if freq < 0.10:
                    to_remove.append(topic)
            for topic in to_remove:
                self.related_topics.discard(topic)
        # No size caps: retain full background_knowledge and related_topics

    def parse_json_strict(self, text: str) -> dict | None:
        """Extract and parse the first JSON object from text."""
        if not text:
            return None
        # Direct parse
        try:
            return json.loads(text)
        except Exception:
            pass
        # Try fenced code block
        try:
            if "```" in text:
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    return json.loads(text[start : end + 1])
        except Exception:
            pass
        # Fallback: find braces region
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
        except Exception:
            return None
        return None

    def _send_request(
        self,
        url: str,
        headers: dict,
        payload: dict,
        stage_name: str,
        model: str,
        timeout_override: float | None = None,
    ) -> tuple[dict, float] | None:
        """Send HTTP request to DeepSeek API with error handling and i18n logs."""
        console.print(
            self.i18n.t("deepseek.sending_request", stage=stage_name, model=model)
        )
        start_t = time.perf_counter()
        resp = None
        data: Any = None
        try:
            with console.status(
                f"[bold blue]{self.i18n.t('deepseek.waiting_response', stage=stage_name)}"
            ):
                timeout = (
                    float(timeout_override)
                    if timeout_override is not None
                    else (120.0 if "reason" in model else 90.0)
                )
                resp = self.client.post(
                    url, json=payload, headers=headers, timeout=httpx.Timeout(timeout)
                )
            elapsed = time.perf_counter() - start_t
            console.print(
                self.i18n.t(
                    "deepseek.request_complete",
                    stage=stage_name.title(),
                    status=resp.status_code,
                    time=f"{elapsed:.2f}",
                )
            )
            resp.raise_for_status()
            data = resp.json()
            self._last_error_kind = None
            return data, elapsed
        except httpx.TimeoutException as e:
            self._last_error_kind = "timeout"
            console.print(
                Panel(
                    self.i18n.t(
                        "deepseek.error_timeout_body",
                        stage=stage_name.title(),
                        error=str(e),
                    ),
                    title=f"[red]{self.i18n.t('deepseek.error_timeout_title')}[/red]",
                    border_style="red",
                )
            )
            return None
        except httpx.RequestError as e:
            self._last_error_kind = "network"
            console.print(
                Panel(
                    self.i18n.t(
                        "deepseek.error_network_body",
                        stage=stage_name.title(),
                        error=str(e),
                    ),
                    title=f"[red]{self.i18n.t('deepseek.error_network_title')}[/red]",
                    border_style="red",
                )
            )
            return None
        except httpx.HTTPStatusError as e:
            self._last_error_kind = "http"
            body = None
            try:
                body = resp.text if resp is not None else None
            except Exception:
                body = None
            console.print(
                Panel(
                    self.i18n.t(
                        "deepseek.error_http_body",
                        stage=stage_name.title(),
                        error=str(e),
                        body=body,
                    ),
                    title=f"[red]{self.i18n.t('deepseek.error_http_title')}[/red]",
                    border_style="red",
                )
            )
            return None
        except Exception as e:
            self._last_error_kind = "unknown"
            console.print(
                Panel(
                    self.i18n.t(
                        "deepseek.error_unknown_body",
                        stage=stage_name.title(),
                        error=str(e),
                    ),
                    title=f"[red]{self.i18n.t('deepseek.error_unknown_title')}[/red]",
                    border_style="red",
                )
            )
            return None

    def close(self):
        """Close HTTP client."""
        if hasattr(self, "client"):
            self.client.close()
