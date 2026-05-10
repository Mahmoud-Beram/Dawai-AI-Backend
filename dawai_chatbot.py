import json
import pandas as pd
import re
import random
from thefuzz import fuzz
import sys
from sentence_transformers import SentenceTransformer, util
import warnings
warnings.filterwarnings(action="ignore", category=FutureWarning)

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')


class DawaiChatbot:
    def __init__(self, db_path=None, herbs_path=None):
        import os
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_dir = os.getcwd()
        self.db_path   = db_path   or os.path.join(base_dir, 'egyptian_medicines.csv')
        self.herbs_path= herbs_path or os.path.join(base_dir, 'alternative_medicine.json')

        # ── Load Databases ───────────────────────────────────────────────────
        self.medicines_df = pd.read_csv(self.db_path)
        try:
            with open(self.herbs_path, 'r', encoding='utf-8') as f:
                self.herbs_db = json.load(f)
                for herb in self.herbs_db:
                    if 'symptoms' in herb:
                        herb['symptoms'] = [self.clean_text(s) for s in herb['symptoms']]
        except Exception:
            self.herbs_db = []

        # Fast-lookup indexes
        self.medicines_df['Name_EN_lower'] = self.medicines_df['Name_EN'].astype(str).str.lower()
        self.medicines_df['Name_AR']       = self.medicines_df['Name_AR'].astype(str)
        self.medicines_df['Uses']          = self.medicines_df['Uses'].astype(str)

        # ── [1] Negation Words ───────────────────────────────────────────────
        # كل الأدوات اللي بتنفي جملة في العامية المصرية
        self.negation_words = [
            "مش", "مو", "لا", "مفيش", "ملوش",
            "ماعنديش", "مش عندي", "مش بشتكي",
            "مش حاسس", "مش فيه", "مش في",
        ]

        # ── [4] Chit-Chat Templates (Expanded & Varied) ──────────────────────
        self.greetings = [
            "أهلاً بيك في صيدلية Dawai!  بتشتكي من إيه النهارده؟",
            "يا هلا! إزاي أقدر أساعدك؟ ",
            "مرحباً بك! أنا المساعد الطبي بتاعك، طمني بتشتكي من إيه؟ ",
            "أهلين وسهلين! كلمني، أنا هنا أساعدك. ",
            "يسعدلي مساك! Dawai معاك دايماً. بتحتاج إيه؟ ",
            "وعليكم السلام ورحمة الله وبركاته! أقدر أساعدك إزاي؟ ",
            "يا هلا بيك يا فندم! طمني صحتك عاملة إيه؟ ",
            "صباح الفل والورد! تحت أمرك، بتشتكي من حاجة؟ "
        ]
        self.farewells = [
            "سلامتك! ربنا يشفيك ويعافيك. ",
            "تصبح على خير! أي وقت تحتاجني أنا هنا. ",
            "مع ألف سلامة! خالي بالك من صحتك. ",
            "يسلم جسمك! لو احتجت حاجة تاني كلمني. ",
            "في رعاية الله! وألف سلامة عليك. ",
            "مع السلامة يا فندم! وربنا يبعد عنك أي مرض. "
        ]
        self.thanks_responses = [
            "العفو! ده واجبي.  لو في أي حاجة تانية، كلمني.",
            "أي خدمة! ربنا يشفيك. ",
            "يسلملك! صحتك أهم حاجة. ",
            "شكراً ليك أنت! لو فضل ألم لا تتردد. ",
            "تحت أمرك في أي وقت! ألف سلامة. ",
            "العفو يا فندم! إحنا هنا عشان راحتك وصحتك. ",
            "حبيبي تسلم! شفاك الله وعافاك. "
        ]
        self.empathy = [
            "ألف سلامة عليك! شفاك الله وعافاك. ",
            "ماتشوفش شر إن شاء الله. ",
            "سلامتك من التعب! دعواتي ليك بالشفاء العاجل. ",
            "الله يشفيك ويعافيك! هنشوف إيه الأنسب ليك. ",
            "ربنا يريحك من الألم ده.  خليني أساعدك.",
            "شفاك الله وعافاك! متقلقش كل حاجة ليها علاج. ",
            "سلامتك ألف سلامة! إن شاء الله تكون بسيطة. "
        ]
        # قوالب تقديم الدواء — بتتغير عشوائياً عشان متبانش آلة
        self.med_intros = [
            " **{reason}** إليك أبرز المقترحات:",
            " **{reason}** ده اللي بنقترحه عليك:",
            " **{reason}** من أفضل الخيارات المتاحة:",
            " **{reason}** جرب الأدوية دي بناءً على وصفك:",
            " **{reason}** دي أقرب أدوية ممكن تفيدك:",
        ]
        self.unknown = [
            "عذراً، مش قادر أفهم قصدك بالظبط. ياريت توضحلي اسم الدواء أو العَرَض اللي بتشتكي منه. ",
            "ممكن توضحلي أكتر؟ بتسأل عن دواء معين ولا عندك عرض طبي (زي صداع أو مغص)؟",
            "مش متأكد فاهم قصدك. ممكن تقولي بتتألم منين بالظبط؟ ",
            "يا ريت تكتبلي العرض اللي بتحس بيه بوضوح أكتر عشان أقدر أفيدك. ",
            "الكلام مش واضح ليا للأسف.. تقدر تقولي بتشتكي من إيه بالظبط؟ "
        ]

        # ── Symptom → DB Keywords Mapping ───────────────────────────────────
        self.symptom_keywords = {
            "صداع":    ["صداع", "ألم الرأس", "للصداع", "والصداع", "نصفي",
                        "دماغي وجعاني", "راسي بتوجعني", "مصدع",
                        "دماغي بتوجعني", "راسي بتنقح", "دماغي هتفرتك",
                        "راسي", "دماغي", "صظاع"],
            "سخونيه": ["سخونية", "حمى", "حراره", "حرارة", "سخن", "مولع",
                        "درجة حرارتي", "جسمي سخن", "سخونيه", "سخونه", "سخنه"],
            "برد":     ["برد", "زكام", "رشح", "انفلونزا", "جيوب أنفية",
                        "عطس", "انف", "عندي برد", "مزكم", "بردان"],
            "تنفس":    ["موسع شعب", "ضيق تنفس", "نهجان", "ربو", "شعب هوائية",
                        "تنفس", "مخنوق مش قادر اتنفس", "مخنوق", "بنهج", "نفسي", "صدري", "الصدر", "صدري واجعني"],
            "غثيان":   ["ترجيع", "رغبه في القيء", "عايز ارجع", "غثيان", "قيء",
                        "نفسي غامة عليا", "قرفان", "برجع", "هرجع"],
            "مغص":     ["مغص", "تقلصات", "المغص", "والتقلصات", "بطني", "معدتي",
                        "بطمي", "بطني وجعاني", "معدتي بتوجعني",
                        "بطني بتتقطع", "وجع في بطني", "امعائي", "معدي"],
            "كحة":     ["كحة", "سعال", "بلغم", "للكحة", "بكح", "كح", "كحه"],
            "اسهال":   ["إسهال", "اسهال", "سهال"],
            "امساك":   ["إمساك", "امساك", "ملين", "ممسك"],
            "عظام":    ["عظمي", "مفاصل", "ركبتي", "الروماتيزم", "عظام", "هيكل",
                        "هشاشة", "ظهري", "ضهري", "ضهر", "الضهر", "ورم", "تورم",
                        "كدمات", "وقعه", "العضلات", "عضلي",
                        "وقعت ورجلي وارمه", "كوره", "رجلي", "تكسير", "مكسر", "همدان"],
            "حساسية":  ["حساسية", "بهرش", "هرش", "حكه", "حكة", "طفح جلدي",
                        "أرتيكاريا", "أكلت حاجة ووشي احمر", "حساسيه", "احمرار"],
            "حروق":    ["حرق", "حروق", "اتحرقت", "محروق", "زيت", "ميه سخنه", "مياه مغليه", "مغلية", "لسعه", "لسعة", "نار"],
            "حموضة":   ["حموضة", "ارتجاع", "حرقان", "حموضه", "حامض", "حرقاني"],
            "سكر":     ["السكر", "مرض السكري", "الدم", "سكر", "سكري"],
            "ضغط":     ["ضغط الدم", "الضغط المرتفع", "ارتفاع ضغط", "ضغط"],
            "قلب":     ["القلب", "الشرايين", "عضلة القلب", "قلبي"],
            "اعصاب":   ["الأعصاب", "التهاب الأعصاب", "ضعف العصب", "اعصاب", "عصب", "اعصابي"],
            "اكتئاب":  ["مضاد للاكتئاب", "اكتئاب", "قلق", "توتر", "مكتئب", "متوتر"],
            "خصوبة":   ["خصوبة", "إنجاب", "حيوانات منوية", "تكيس", "تبويض"],
            "التهاب":  ["مضاد حيوي", "بكتيريا", "التهاب", "ميكروب", "خراج"],
            "مسالك":   ["المسالك", "المثانة", "البروستاتا", "التبول",
                        "حمام", "بول", "كلى", "حرقان ف البول"],
            "مرارة":   ["المرارة", "حصوات", "الكبد", "مرارة", "صفرا"],
            "فيتامين": ["فيتامين", "مكمل غذائي", "معادن", "إرهاق", "نقص",
                        "هبوط", "دايخ", "بدوخ", "انيميا", "دوخة"],
            "تخسيس":   ["تخسيس", "السمنة", "فقدان الوزن", "حرق دهون", "دايت", "وزني", "تخين"],
            "شعر":     ["تساقط الشعر", "الصلع", "قشرة", "لشعر", "شعري", "صلع", "شعر", "قشره"],
            "بشرة":    ["البشرة", "حب الشباب", "تجاعيد", "غسول", "تفتيح",
                        "بشرة", "بشره", "وشي", "جلد"],
            "قطرة":    ["قطرة", "جفاف العين", "المياه الزرقاء", "حمراء",
                        "عيني", "ودني", "قطره", "أذن", "اذن", "وداني"],
            "اسنان":   ["اسناني", "سنتي", "أسناني", "اسنان", "ضرسي", "ضرس", "لثة", "لثه"],
        }

        self.symptom_responses = {
            "صداع":    "مناسب جداً كمسكن سريع لتخفيف الصداع وألم الرأس.",
            "سخونيه":  "بيشتغل كخافض ممتاز للحرارة العالية وبيسكن وجع الجسم.",
            "برد":     "بيخفف من أعراض نزلات البرد، الرشح، وتكسير الجسم.",
            "غثيان":   "بيهدي جدار المعدة وبيمنع الإحساس بالرغبة في الترجيع.",
            "مغص":     "بيعمل كمهدئ قوي لتقلصات المعدة وعلاج المغص.",
            "كحة":     "مهدئ فعال للسعال المستمر وبيساعد في طرد البلغم.",
            "اسهال":   "مطهر معوي بيعالج حالات الإسهال وينظم حركة الأمعاء.",
            "امساك":   "ملين لطيف بيسهل عملية الهضم وبيعالج الإمساك.",
            "عظام":    "مسكن قوي ومضاد لالتهابات العظام، بيريح وجع المفاصل والضهر.",
            "حساسية":  "مضاد هيستامين بيقلل الحساسية، الهرش، وتهيج الجلد.",
            "حموضة":   "بيعادل حمض المعدة بسرعة وبيطفي حرقان الصدر.",
            "سكر":     "بيساعد في تظبيط مستوى السكر في الدم.",
            "ضغط":     "بيستخدم لمعادلة والتحكم في ضغط الدم المرتفع.",
            "قلب":     "بيدعم وظايف عضلة القلب وبيوسع الأوعية الدموية.",
            "اعصاب":   "بيقوي الأعصاب الضعيفة وبيعالج التهاباتها.",
            "اكتئاب":  "بيحسن الحالة المزاجية وبيقلل من التوتر العصبي.",
            "خصوبة":   "مكمل طبي مميز لدعم الصحة الإنجابية والخصوبة.",
            "التهاب":  "مضاد حيوي قوي بيقضي على مسببات البكتيريا والالتهاب.",
            "مسالك":   "بيخفف من التهابات وحرقان المسالك البولية وتطهيرها.",
            "مرارة":   "بينشط وظايف الكبد وبيحسن من إنتاج العصارة الصفراوية.",
            "فيتامين": "مكمل غذائي غني بيرد الفيتامينات الناقصة وبيدي طاقة للجسم.",
            "تخسيس":   "عامل مساعد آمن في برامج فقدان الوزن وحرق الدهون.",
            "شعر":     "بيغذي بصيلات الشعر من الفروة وبيقلل التساقط المقلق.",
            "بشرة":    "منتج لطيف مخصص للعناية بصحة البشرة ونضارتها.",
            "قطرة":    "قطرة ملطفة للعين/الأذن بتخفف الجفاف أو التورم.",
            "تنفس":    "موسع للشعب الهوائية بيسهل التنفس وبيفك الخنقة.",
            "اسنان":   "مسكن ومضاد التهاب موضعي لتخفيف آلام الأسنان واللثة.",
        }

        # ── Load NLP Model ───────────────────────────────────────────────────
        print("🤖 [Chatbot System] Loading Semantic NLP Model")
        try:
            # نستخدم نفس base_dir المعرّف فوق بدل إعادة تعريفه غلط
            local_model_path = os.path.join(base_dir, 'models', 'camelbert_model')
            if os.path.exists(local_model_path):
                print("🔄 Loading offline CAMeL-BERT model from local disk...")
                self.nlp_model = SentenceTransformer(local_model_path)
            else:
                print("⏳ Downloading CAMeL-BERT model for the first time (~400MB)...")
                self.nlp_model = SentenceTransformer('CAMeL-Lab/bert-base-arabic-camelbert-mix')

            with open(os.path.join(base_dir, 'egyptian_symptom_sentences.json'), 'r', encoding='utf-8') as f:
                self.symptom_sentences = json.load(f)

            self.target_nlp_sentences = []
            self.target_nlp_mapping   = []
            for sym, sentences in self.symptom_sentences.items():
                for sentence in sentences:
                    self.target_nlp_sentences.append(sentence)
                    self.target_nlp_mapping.append(sym)

            self.db_embeddings = self.nlp_model.encode(self.target_nlp_sentences)
            print(f"✅ [Chatbot System] NLP Model loaded — {len(self.target_nlp_sentences)} sentences indexed.")
        except Exception as e:
            print(f" [Chatbot System] NLP Model failed: {e}. Falling back to fuzzy logic.")
            self.nlp_model    = None
            self.db_embeddings = None

    # ════════════════════════════════════════════════════════════════════════
    # [1] ADVANCED TEXT NORMALIZATION
    # ════════════════════════════════════════════════════════════════════════
    def clean_text(self, text: str) -> str:
        """
        تطبيع شامل للنص العربي المصري:
          1. إزالة علامات الترقيم
          2. إزالة كل التشكيل (فتحة، ضمة، كسرة، شدة ...)
          3. توحيد الهمزات  (أ إ آ ٱ) ← ا
          4. توحيد التاء المربوطة  ة ← ه
          5. توحيد الألف المقصورة  ى ← ي
          6. تطبيع الهمزات الطرفية  ئ ← ي  /  ؤ ← و
          7. تكسير المط في الحروف  "صداااع" ← "صداع"
        """
        text = re.sub(r'[^\w\s]', '', text)                      # [1] ترقيم
        text = re.sub(r'[\u064B-\u065F\u0670]', '', text)        # [2] تشكيل
        text = re.sub(r'[أإآٱ]', 'ا', text)                      # [3] همزات
        text = re.sub(r'ة', 'ه', text)                            # [4] تاء مربوطة
        text = re.sub(r'ى', 'ي', text)                            # [5] ألف مقصورة
        text = re.sub(r'ئ', 'ي', text)                            # [6] همزة على ياء
        text = re.sub(r'ؤ', 'و', text)                            # [6] همزة على واو
        text = re.sub(r'(.)\1{2,}', r'\1', text)                # [7] مط → حرف واحد  "صداااع" ← "صداع"
        return text.strip().lower()

    # ════════════════════════════════════════════════════════════════════════
    # [2] NEGATION DETECTION
    # ════════════════════════════════════════════════════════════════════════
    def is_negated(self, msg: str, keyword: str) -> bool:
        """
        بتشوف لو الكلمة دي مسبوقة بأداة نفي في نطاق 3 كلمات قبلها.
        مثال: "أنا مش عندي صداع" ← يرجع True للـ keyword "صداع"
        """
        words = msg.split()
        for i, word in enumerate(words):
            if keyword in word:
                # نافذة الـ 3 كلمات اللي قبل الكلمة دي
                window = words[max(0, i - 3):i]
                window_text = ' '.join(window)
                if any(neg in window_text for neg in self.negation_words):
                    return True
        return False

    # ════════════════════════════════════════════════════════════════════════
    # MEDICAL INTENT DETECTION (unchanged)
    # ════════════════════════════════════════════════════════════════════════
    def is_medical_query(self, text: str) -> bool:
        medical_triggers = [
            "عندي", "بشتكي", "بيوجعني", "واجعني", "تعبان", "مريض", "حاسس", "حاسه",
            "دواء", "علاج", "عقار", "دكتور", "صيدلية", "برشام", "حقنه",
            "اعراض", "مرض", "الم", "وجع", "حراره", "صداع", "مغص", "كحه", "زكام",
            "حبوب", "وشي", "دم", "ضغط", "سكر", "حرقان", "حرقاني", "حموضه", "اسهال",
            "امساك", "عضمي", "رجلي", "دماغي", "بطني", "معدتي", "قلبي", "دوخه", "دايخ",
            "صدري", "صدر", "ضهري", "ضهر", "عيني", "ودني", "سنتي", "ضرسي",
            "حرق", "حروق", "اتحرقت", "محروق", "زيت", "مغليه", "سخنه", "نار", "لسعه"
        ]
        return any(trigger in text for trigger in medical_triggers)

    # ════════════════════════════════════════════════════════════════════════
    # MAIN MESSAGE PROCESSOR
    # ════════════════════════════════════════════════════════════════════════
    def process_message(self, user_message: str, context_medicine=None) -> dict:
        msg_cleaned = self.clean_text(user_message)

        # Guard: رسالة فاضية أو علامات بس
        if len(msg_cleaned) < 2:
            return {
                "text": "عذراً، لم أفهم رسالتك. هل يمكنك توضيح سؤالك؟",
                "context_medicine": context_medicine,
            }

        # ── [4] Chit-Chat Detection ──────────────────────────────────────────
        # تحية (بنتحقق منها الأول عشان "السلام عليكم" متتحسبش "سلام" بتاعت وداع)
        if any(w in msg_cleaned for w in ["السلام", "سلام عليكم", "اهلا", "أهلا", "مرحبا", "ازيك", "عامل ايه", "عامله ايه", "ايه الاخبار", "صباح", "مسا"]):
            return {"text": random.choice(self.greetings), "context_medicine": None}

        # وداع
        if any(w in msg_cleaned for w in ["باي", "مع السلامه", "مع السلامة", "وداعا", "تصبح على خير", "اقفل", "سلام"]):
            return {"text": random.choice(self.farewells), "context_medicine": None}

        # شكر
        if any(w in msg_cleaned for w in ["شكرا", "شكراً", "ميرسي", "تسلم", "تسلمي", "يسلمو", "الف شكر", "يعطيك العافيه", "جزاك الله", "كتر خيرك"]):
            return {"text": random.choice(self.thanks_responses), "context_medicine": context_medicine}

        response_parts = []
        is_sickness    = False

        # ── Safety Layer 1: EMERGENCY ────────────────────────────────────────
        emergency_keywords = [
            "انزف", "بنزف", "نزيف", "جلطه", "مش قادر اتنفس",
            "قلبي بيوجعني", "اغمي", "فقدت الوعي", "الم في الصدر", "سم", "تسمم",
        ]
        if any(k in msg_cleaned for k in emergency_keywords):
            return {
                "text": (
                    " **تحذير طبي طارئ:** الأعراض التي تصفها خطيرة وقد تشير إلى حالة طوارئ. "
                    "يرجى التوجه لأقرب مستشفى أو الاتصال بالإسعاف فوراً. "
                    "يُمنع ترشيح أدوية في الحالات الحرجة حرصاً على حياتك."
                ),
                "context_medicine": None,
            }

        # ── Safety Layer 2: PREGNANCY ────────────────────────────────────────
        if any(k in msg_cleaned for k in ["حامل", "حمل", "مراتي حامل", "زوجتي حامل", "حوامل"]):
            return {
                "text": (
                    " **تنبيه طبي خاص بالحمل:** نظراً لوجود حمل، يُمنع تماماً تناول أي أدوية "
                    "دون استشارة طبيب النساء والتوليد المتابع للحالة، حيث أن العديد من الأدوية "
                    "قد تسبب تشوهات للجنين أو مضاعفات للحمل. يرجى مراجعة الطبيب."
                ),
                "context_medicine": None,
            }

        # ── Safety Layer 3: SEVERITY ─────────────────────────────────────────
        severity_keywords = ["شديد", "بموت", "مش قادر", "فظيع", "قوي جدا", "مش مستحمل"]
        is_severe = any(k in msg_cleaned for k in severity_keywords)

        # ── Contextual Follow-up (Side Effects / Uses) ───────────────────────
        if context_medicine:
            ctx = context_medicine.strip().lower()
            med_filter = (
                (self.medicines_df['Name_EN'].str.strip().str.lower() == ctx) |
                (self.medicines_df['Name_AR'].str.strip() == context_medicine.strip())
            )
            if any(w in msg_cleaned for w in ["اعراض", "اضرار", "جانبيه", "سلبيات", "موانع"]):
                med_info = self.medicines_df[med_filter]
                if not med_info.empty:
                    return {
                        "text": (
                            f"بالنسبة لدواء **{context_medicine}**، الأعراض الجانبية هي:\n"
                            f"{med_info.iloc[0]['SideEffects']}\n\n"
                            "* ملاحظة طبية: يجب استشارة الطبيب أو الصيدلي دائماً.*"
                        ),
                        "context_medicine": context_medicine,
                    }
            if any(w in msg_cleaned for w in ["استخدام", "ليه", "دواعي", "بيعمل ايه", "فايدته"]):
                med_info = self.medicines_df[med_filter]
                if not med_info.empty:
                    return {
                        "text": (
                            f"دواء **{context_medicine}** يُستخدم في:\n"
                            f"{med_info.iloc[0]['Uses']}\n\n"
                            "* ملاحظة طبية: يجب الرجوع للطبيب لتحديد الجرعة المناسبة.*"
                        ),
                        "context_medicine": context_medicine,
                    }

        # ── Dose & Usage Query Detection (الجرعة) ────────────────────────────
        if any(w in msg_cleaned for w in ["جرعه", "جرعة", "جرعته", "جرعاته", "الجرعات", "كام مره", "كم مرة", "اخدها ازاي", "اخده ازاي", "الجرعه", "الجرعة", "حبايه", "قرص", "كام سم"]):
            return {
                "text": (
                    " بالنسبة لتحديد الجرعة (كم مرة أو كم حبة)، ده بيعتمد على السن والوزن والتاريخ المرضي. "
                    "الأفضل والأأمن إنك تستشير الصيدلي أو الطبيب المعالج عشان يحددلك الجرعة المظبوطة."
                ),
                "context_medicine": context_medicine,
            }

        # ── Medicine Name Lookup ─────────────────────────────────────────────
        # BUG FIX: نبحث عن اسم دواء بس لو الرسالة قصيرة (≤ 5 كلمات)
        # الشرط القديم كان "or len > 3" اللي كان صح دايماً وبيكسر الـ symptom flow
        best_med_match = None
        best_med_score = 0

        if len(msg_cleaned.split()) <= 5:
            scores_en = self.medicines_df['Name_EN_lower'].apply(
                lambda x: fuzz.partial_token_set_ratio(msg_cleaned, x) if len(x) >= 3 else 0
            )
            scores_ar = self.medicines_df['Name_AR'].apply(
                lambda x: fuzz.partial_token_set_ratio(msg_cleaned, x) if len(x) >= 3 else 0
            )
            combined   = scores_en.combine(scores_ar, max)
            best_idx   = combined.idxmax()
            best_med_score = combined[best_idx]
            best_med_match = self.medicines_df.loc[best_idx] if best_med_score > 0 else None

        if best_med_match is not None and best_med_score >= 95:
            row = best_med_match
            return {
                "text": (
                    f"**{row['Name_EN']} ({row['Name_AR']}):**\n"
                    f" **دواعي الاستعمال:** {row['Uses']}\n"
                    f" **الأعراض الجانبية:** {row['SideEffects']}\n\n"
                    "* تنبيه: هذه معلومات إرشادية، ولا تغني عن استشارة الطبيب المُعالج.*"
                ),
                "context_medicine": row['Name_EN'],
            }

        # ════════════════════════════════════════════════════════════════════
        # [3] MULTI-SYMPTOM DETECTION
        # بدل ما نوقف على أول match، نلم كل الأعراض في الرسالة
        # ════════════════════════════════════════════════════════════════════
        matched_symptoms = []   # list of (symptom_name, db_keywords)
        user_words       = msg_cleaned.split()

        # LEVEL 1 — Exact / Fuzzy Matching
        for sym, db_keys in self.symptom_keywords.items():
            clean_sym  = self.clean_text(sym)
            clean_keys = [self.clean_text(k) for k in db_keys]

            matched = False
            # 1a. Exact substring
            if clean_sym in msg_cleaned or any(k in msg_cleaned for k in clean_keys):
                matched = True
            else:
                # 1b. Fuzzy (يتحمل تيبوهات زي "ضري" بدل "ضهري")
                for u_word in user_words:
                    if len(u_word) < 4:
                        continue
                    for k in clean_keys:
                        req_score = 90 if len(k) <= 5 else 85
                        if fuzz.ratio(u_word, k) >= req_score:
                            matched = True
                            break
                    if matched:
                        break

            # [2] تطبيق Negation Detection — لو العرض منفي نتجاهله
            if matched and not self.is_negated(msg_cleaned, clean_sym):
                matched_symptoms.append((sym, db_keys))
                is_sickness = True

        # LEVEL 2 — Semantic NLP (لو Level 1 ما لاقاش حاجة)
        if not matched_symptoms and self.nlp_model is not None and self.db_embeddings is not None \
                and len(msg_cleaned.strip()) > 3:
            user_embedding = self.nlp_model.encode(msg_cleaned)
            similarities   = util.cos_sim(user_embedding, self.db_embeddings)[0]
            best_idx       = similarities.argmax().item()
            best_score     = similarities[best_idx].item()
            best_sym       = self.target_nlp_mapping[best_idx]
            is_medical     = self.is_medical_query(msg_cleaned)

            if best_score >= 0.82 and is_medical:
                matched_symptoms.append((best_sym, self.symptom_keywords.get(best_sym, [])))
                is_sickness = True
            elif 0.75 <= best_score < 0.82 and is_medical:
                return {
                    "text": (
                        f"أعتقد أنك تقصد أعراض متعلقة بـ **{best_sym}**، صح كده؟\n"
                        f"لو ده قصدك، ياريت تكتبلي (عندي {best_sym}) عشان أتأكد وأقدر أجبلك العلاج المناسب. "
                    ),
                    "context_medicine": context_medicine,
                }

        # ── Build Response ───────────────────────────────────────────────────
        if is_sickness:
            response_parts.append(random.choice(self.empathy))

        is_pediatric = any(w in msg_cleaned for w in
                           ["طفل", "اطفال", "اطفال", "رضيع", "ابني", "بنتي", "عيالي", "عيل"])

        suggested_medicine = None

        # نعالج العرض الأول كأولوية — وننوّه بالباقي
        if matched_symptoms:
            primary_sym, primary_keys = matched_symptoms[0]
            extra_syms = [s[0] for s in matched_symptoms[1:]]

            # [3] لو في أعراض تانية نوّه بيها
            if extra_syms:
                response_parts.append(
                    f"_لاحظت كمان إنك بتشتكي من: {', '.join(extra_syms)} — "
                    f"خليني أعالج **{primary_sym}** الأول._ "
                )

            simple_reason = self.symptom_responses.get(primary_sym, "دواء ممتاز ومناسب لحالتك.")
            if is_pediatric:
                simple_reason += " (آمن وملائم للأطفال - يرجى مراجعة الجرعة)"

            # Vectorized filtering
            uses_col    = self.medicines_df['Uses'].astype(str)
            name_ar_col = self.medicines_df['Name_AR'].astype(str)
            keyword_mask = uses_col.apply(lambda u: any(k in u for k in primary_keys))

            if is_pediatric:
                ped_name = name_ar_col.apply(
                    lambda n: any(k in n for k in ["شراب", "نقط", "أطفال", "للاطفال", "عصير"])
                )
                ped_uses = uses_col.apply(
                    lambda u: any(k in u for k in ["أطفال", "للاطفال", "الرضع"])
                )
                keyword_mask = keyword_mask & (ped_name | ped_uses)

            filtered = self.medicines_df[keyword_mask]
            all_matching_meds = []
            for _, row in filtered.iterrows():
                all_matching_meds.append((
                    f" **{row['Name_AR']}**: (قسم {str(row['Category']).strip()})",
                    row['Name_EN'],
                ))

            if all_matching_meds:
                # [Previous Bug Fix] نرجع 3 خيارات بدل 1
                selected_meds  = random.sample(all_matching_meds, min(3, len(all_matching_meds)))
                suggested_medicine = selected_meds[0][1]

                # [4] استخدم قالب عشوائي من med_intros
                intro = random.choice(self.med_intros).format(reason=simple_reason)
                response_parts.append(f"\n{intro}")
                response_parts.extend([s[0] for s in selected_meds])

            # Herbs
            matching_herbs = [
                f" **{h['herb']}**: {h['method']} ({h['benefits']})"
                for h in self.herbs_db
                if primary_sym in h.get('symptoms', [])
            ]
            if matching_herbs:
                response_parts.append("\n**الطب البديل والأعشاب الطبيعية (خيار آمن):**")
                response_parts.append(random.choice(matching_herbs))
                response_parts.append(" الطبيعة دايماً فيها الشفاء كخطوة أولى!")

        if response_parts:
            if is_severe:
                response_parts.append(
                    "\n **تنبيه لشدة الألم:** نظراً لأنك تعاني من ألم شديد، "
                    "إذا لم تشعر بتحسن خلال 24 ساعة يجب زيارة الطبيب فوراً لتفادي المضاعفات."
                )
            else:
                response_parts.append(
                    "\n* إخلاء مسؤولية: يرجى استشارة الطبيب أو الصيدلي دائماً قبل تناول أي أدوية.*"
                )
            return {
                "text": "\n".join(response_parts),
                "context_medicine": suggested_medicine,
            }

        # Fallback
        return {"text": random.choice(self.unknown), "context_medicine": None}


# ── Lazy Singleton (لا نعمل init وقت الـ import) ───────────────────────────
_bot_instance = None

def _get_bot():
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = DawaiChatbot()
    return _bot_instance

def get_chat_response(message: str, context=None) -> dict:
    return _get_bot().process_message(message, context)


if __name__ == "__main__":
    print(get_chat_response("عندي مغص شديد ومش قادر"))
    print("=" * 40)
    print(get_chat_response("معلومات عن عقار كونكور"))
    print("=" * 40)
    print(get_chat_response("الاعراض الجانبيه لده", context="Panadol Extra 500mg Tablets"))
