# spec.md — AI Usage Disclosure

This file is the disclosure artifact required by the task's "AI Usage Disclosure"
section. An AI coding assistant (Claude) was used to implement this project's
Registry / Discovery / Router / Planner / Orchestrator architecture. Below is the
technical design document given to the assistant verbatim, which the assistant was
asked to implement as closely as possible, with implementation-level decisions
(where the document didn't specify a concrete mechanism) documented in README.md's
"Assumptions" section instead of silently guessed at.

Notable implementation decisions made by the assistant that are *not* fully specified
in the document below (see README.md "Assumptions" for the full reasoning on each):
- Skill Discovery's embedding/similarity backend is scikit-learn TF-IDF + cosine
  similarity (a free, offline, dependency-light stand-in for a neural embedding
  model), behind a small `EmbeddingIndex` interface so it can be swapped later.
- Every skill shares one generic `SkillInput(text=...)` / `SkillOutput(output=...)`
  contract, rather than each skill defining a bespoke schema, so the Orchestrator and
  Planner can stay fully generic.
- The Planner skips its LLM call entirely for single-skill requests (trivial
  one-step plan) and only invokes the LLM to decide sequential-vs-parallel ordering
  when exactly 2 skills were selected.
- If the Planner's LLM call or plan validation fails, the system falls back to a
  deterministic parallel (non-chained) 2-step plan rather than crashing.

---

## Original document (as provided)

> Technical Design Document — Multi-Skill Agent

### 1. هدف پروژه

هدف پروژه پیاده‌سازی یک Multi-Skill Agent مبتنی بر LangGraph است که بتواند درخواست
طبیعی کاربر را دریافت کند، Skillهای موردنیاز را تشخیص دهد و حداکثر دو Skill را برای
پاسخ به درخواست اجرا کند.

چهار Skill اولیه پروژه عبارت‌اند از: Summarizer، Translator، Calculator، General Chat.

معماری باید Extensible باشد؛ اضافه کردن Skillهای جدید نباید مستلزم تغییر در هسته
Agent، Router یا Orchestrator باشد — حتی اگر تعداد Skillها از ۴ به ۳۰ یا بیشتر
افزایش پیدا کند.

در هر درخواست، Agent حداکثر مجاز به انتخاب دو Skill است.

### 2. نیازمندی‌های اصلی

**2.1 Skillها** — هر Skill یک قابلیت مستقل و قابل اجراست و باید یک قرارداد مشترک
داشته باشد. Skillها نباید منطق یکدیگر را بدانند و Orchestrator نیز نباید برای هر
Skill منطق اختصاصی داشته باشد.

**2.2 Routing** — درخواست خام کاربر ابتدا باید تحلیل شود تا مشخص شود کدام Skillها
مرتبط‌اند. خروجی Routing باید Structured باشد و با Pydantic اعتبارسنجی شود. چون تعداد
Skillها می‌تواند افزایش پیدا کند، Router نباید تمام Skillها را مستقیماً در Prompt خود
دریافت کند.

**2.3 پشتیبانی از تعداد زیاد Skill** — با استفاده از Skill Discovery (Embedding) از
بین Registry، Top-K Candidate Skill پیدا می‌شود و فقط آن‌ها به Router داده می‌شوند، تا
LLM مجبور نباشد با تعریف هر N Skill سروکار داشته باشد.

### 3. معماری کلی

```
                         ┌──────────────────┐
                         │  Skill Registry  │
                         │   Skill 1 ... N  │
                         └────────┬─────────┘
                                  │
                                  ▼
User Prompt ───────────────► Skill Discovery
                                  │
                                  │ Top-K Candidates
                                  ▼
                              Router
                                  │
                                  │ Selected Skills
                                  ▼
                              Planner
                                  │
                                  │ Execution Plan
                                  ▼
                             Validator
                                  │
                                  ▼
                           Orchestrator
                            /          \
                           ▼            ▼
                       Skill 1        Skill 2
                            \          /
                             ▼        ▼
                            Final Response
```

### 4. Skill Registry

مرجع مرکزی تمام Skillهای موجود (name, description, input schema, output schema,
implementation). وظیفه تصمیم‌گیری ندارد. `Registry ≠ Router`, `Registry ≠ Discovery`.

### 5. Skill Discovery

برای هر Skill، description آن به Embedding تبدیل و در Vector Index ذخیره می‌شود. در
زمان اجرا: User Prompt → Query Embedding → Vector Similarity Search → Top-K Skills.
با اضافه شدن Skill جدید، فقط کافی است در Registry ثبت و Embedding آن ایجاد شود؛ هسته
Router و Orchestrator تغییر نمی‌کند.

### 6. Router

تصمیم‌گیری درباره Skillهای موردنیاز از میان Candidateهای Discovery. خروجی Structured:
`class RoutingDecision(BaseModel): skills: list[str]` با محدودیت
`1 ≤ number of selected skills ≤ 2` که در این مرحله enforce می‌شود؛ خروجی نامعتبر باید
Validation/Retry شود.

### 7. Planner

Router مشخص می‌کند چه Skillهایی لازم‌اند؛ Planner مشخص می‌کند این Skillها چگونه باید
اجرا شوند (Sequential وقتی خروجی یک Skill ورودی Skill دیگر است، یا Parallel وقتی دو
عملیات مستقل‌اند). Planner مسئول ترتیب اجرا، Dependency، Parallel/Sequential بودن، و
نحوه انتقال خروجی یک Skill به Skill دیگر است.

### 8. Plan

خروجی Planner باید Structured باشد، شامل steps با id، skill، input (که می‌تواند شامل
`$step_1.output` به‌معنای ارجاع به خروجی Step قبلی باشد) و در صورت نیاز `depends_on`.
Plan نباید مستقیماً بدون اعتبارسنجی اجرا شود.

### 9. Plan Validation

**Structural Validation** (آیا JSON معتبر است؟ fieldها صحیح‌اند؟ حداکثر ۲ Step؟ نوع
داده‌ها صحیح است؟) در برابر **Semantic Validation** (آیا Skill واقعاً در Registry وجود
دارد؟ آیا input با input_schema سازگار است؟ آیا dependency به Step موجود اشاره
می‌کند؟ آیا dependency cycle وجود دارد؟). `LLM → JSON → Pydantic → Semantic Validation
→ Valid Plan`. LLM فقط پیشنهاد می‌دهد؛ سیستم تصمیم نهایی را می‌گیرد.

### 10. Orchestrator

مسئول اجرای Plan معتبر. نباید بداند Calculator یا Translator چگونه کار می‌کنند؛ فقط
باید بتواند یک Skill را از Registry دریافت کرده و `skill.run(input_data)` را فراخوانی
کند. نباید `if skill == "calculator": ...` داشته باشیم؛ بلکه
`Plan → Orchestrator → Registry.get(skill_name) → skill.run(...)`.

### 11. Calculator و Tool Calling

LLM نباید خودش نتیجه محاسبه را حدس بزند؛ فقط expression را استخراج می‌کند و Calculator
Tool محاسبه واقعی را انجام می‌دهد. برای اجرای امن expression از روش محدود و
کنترل‌شده استفاده می‌کنیم، نه `eval()` ناامن مستقیم.

### 12. Fallback

اگر هیچ Skill مناسبی پیدا نشود یا درخواست مبهم باشد: `No relevant skill → General
Chat`. General Chat همچنین برای پرسش‌های عمومی که متعلق به Summarizer، Translator یا
Calculator نیستند استفاده می‌شود.

### 13. پشتیبانی از فارسی

تمام لایه‌هایی که با زبان طبیعی کار می‌کنند (Embedding Model، Skill Discovery، Router
LLM، خروجی) باید توانایی مناسبی در پردازش فارسی داشته باشند.

### 14. LLM Provider

Free-tier، حداکثر 35B پارامتر، در README مدل و تعداد پارامترهای آن ذکر شود (مثلاً
OpenRouter). LLM نقش Routing / Planning / Input Extraction / Text Generation دارد؛
اجرای عملیات محاسباتی را به Calculator Tool واگذار می‌کند.

### 15. LangGraph

کل Workflow باید با LangGraph پیاده‌سازی شود: `Discovery Node → Router Node → Planner
Node → Validation → Execution Node(s) → Final Response`. حداقل Routing Node و
Execution Node(ها) باید در Graph وجود داشته باشند. LangGraph مسئول کنترل جریان اجرای
Agent است، نه اجرای منطق داخلی Skillها.

### 16. State

`AgentState`: user_prompt, candidate_skills, routing_decision, plan,
execution_results, final_response. هر Node بخش مربوط به خودش را به‌روزرسانی می‌کند.

### 17. Extensibility

> Adding a new Skill should not require modifying the core Agent logic.

Skillها به‌صورت Plugin-like طراحی می‌شوند؛ هسته سیستم به نام و قرارداد Skill وابسته
است، نه implementation داخلی آن: `New Skill → Register → Create Embedding →
Discovery → Router → Planner → Orchestrator`.

### 18. ساختار پیشنهادی پروژه

```
multi-skill-agent/
├── src/
│   ├── skills/{base.py, calculator.py, translator.py, summarizer.py, general_chat.py}
│   ├── registry/skill_registry.py
│   ├── discovery/skill_discovery.py
│   ├── routing/{router.py, planner.py}
│   ├── schemas/{skill.py, plan.py}
│   ├── orchestration/orchestrator.py
│   ├── graph/graph.py
│   ├── llm/client.py
│   └── main.py
├── tests/
├── README.md
├── spec.md
├── .env
├── .gitignore
└── requirements.txt
```

`spec.md` برای Requirement مربوط به AI Usage Disclosure استفاده می‌شود.

### 19. جریان کامل یک Request

مثال سکوئنشیال («این متن را خلاصه کن و بعد به انگلیسی ترجمه کن»):
`Discovery → Candidate Skills [summarizer, translator, general_chat] → Router →
[summarizer, translator] → Planner → Step 1: Summarizer → Summary Output → Step 2:
Translator(Summary Output) → Final Output`.

مثال مستقل («۲۵ × ۳۷ را حساب کن و این متن را خلاصه کن»): Router → `[calculator,
summarizer]` → Planner تشخیص می‌دهد مستقل‌اند → اجرای موازی → Final Response.

### 20. اصول طراحی نهایی

پنج اصل: **۱) Discovery** — پیدا کردن Skillهای مرتبط بدون قرار دادن همه در Context.
**۲) Routing** — انتخاب Skillهای موردنیاز. **۳) Planning** — تعیین نحوه/ترتیب اجرای
حداکثر دو Skill. **۴) Validation** — جلوگیری از اجرای Plan نامعتبر. **۵) Generic
Execution** — اجرای Skillها بدون Hard-code کردن منطق آن‌ها.
