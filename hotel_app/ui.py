from __future__ import annotations


class HTMLTemplates:
    @staticmethod
    def result_card(title: str, copy: str, css_class: str) -> str:
        return f"""
        <div class="live-result {css_class}">
            <h3>{title}</h3>
            <p>{copy}</p>
        </div>
        """


class CSSTemplates:
    @staticmethod
    def wave_card_hint() -> str:
        return ".live-result { overflow: hidden; position: relative; }"
