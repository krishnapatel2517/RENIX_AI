"""
RENIX Education — Marathi Subject Engine

File:
    RENIX/education/subjects/marathi.py

Purpose:
    Marathi learning, grammar practice, vocabulary,
    comprehension, translation, literature, writing,
    question generation, revision and answer checking.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Iterable


# ============================================================
# DATA MODELS
# ============================================================


@dataclass
class MarathiTopic:
    topic_id: str
    name: str
    category: str
    description: str
    difficulty: str = "medium"
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "difficulty": self.difficulty,
            "keywords": list(self.keywords),
        }


@dataclass
class MarathiQuestion:
    question_id: str
    topic: str
    category: str
    question: str
    answer: Any
    options: list[str] = field(default_factory=list)
    difficulty: str = "medium"
    explanation: str = ""
    hint: str = ""
    marks: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "topic": self.topic,
            "category": self.category,
            "question": self.question,
            "answer": self.answer,
            "options": list(self.options),
            "difficulty": self.difficulty,
            "explanation": self.explanation,
            "hint": self.hint,
            "marks": self.marks,
        }


# ============================================================
# MARATHI ENGINE
# ============================================================


class MarathiEngine:

    def __init__(
        self,
        *,
        seed: int | None = None,
    ) -> None:

        self.random = random.Random(seed)

        self.topics: dict[str, MarathiTopic] = {}

        self.question_counter = 0

        self._register_topics()

    # ========================================================
    # TOPICS
    # ========================================================

    def _register_topics(self) -> None:

        topics = [

            MarathiTopic(
                "grammar",
                "व्याकरण",
                "भाषा",
                "मराठी भाषेतील शब्दरचना, वाक्यरचना आणि व्याकरणाचे नियम.",
                "medium",
                ["शब्द", "वाक्य", "व्याकरण"],
            ),

            MarathiTopic(
                "sandhi",
                "संधी",
                "व्याकरण",
                "दोन वर्ण किंवा शब्द एकत्र आल्यावर होणाऱ्या बदलांचा अभ्यास.",
                "hard",
                ["स्वरसंधी", "व्यंजनसंधी"],
            ),

            MarathiTopic(
                "samas",
                "समास",
                "व्याकरण",
                "दोन किंवा अधिक शब्द एकत्र येऊन तयार होणाऱ्या संक्षिप्त शब्दरचनेचा अभ्यास.",
                "hard",
                ["द्वंद्व", "तत्पुरुष", "बहुव्रीही"],
            ),

            MarathiTopic(
                "alankar",
                "अलंकार",
                "व्याकरण",
                "भाषेला सौंदर्य आणि प्रभाव देणाऱ्या अलंकारांचा अभ्यास.",
                "medium",
                ["उपमा", "रूपक", "अनुप्रास"],
            ),

            MarathiTopic(
                "kavita",
                "कविता",
                "साहित्य",
                "मराठी कवितेचे आकलन, रसग्रहण आणि काव्यवैशिष्ट्यांचा अभ्यास.",
                "medium",
                ["काव्य", "रसग्रहण", "भावार्थ"],
            ),

            MarathiTopic(
                "prose",
                "गद्य",
                "साहित्य",
                "मराठी गद्यलेखन, आकलन आणि आशयाचा अभ्यास.",
                "medium",
                ["गद्य", "आशय", "लेख"],
            ),

            MarathiTopic(
                "literature",
                "मराठी साहित्य",
                "साहित्य",
                "मराठी साहित्याच्या विविध प्रकारांचा आणि साहित्यिकांचा अभ्यास.",
                "medium",
                ["कथा", "कविता", "नाटक", "लेख"],
            ),

            MarathiTopic(
                "vocabulary",
                "शब्दसंपत्ती",
                "भाषा",
                "शब्दांचे अर्थ, समानार्थी, विरुद्धार्थी आणि योग्य उपयोग.",
                "easy",
                ["समानार्थी", "विरुद्धार्थी", "शब्दार्थ"],
            ),

            MarathiTopic(
                "idioms",
                "वाक्प्रचार",
                "भाषा",
                "मराठीतील वाक्प्रचारांचे अर्थ आणि वाक्यातील उपयोग.",
                "medium",
                ["वाक्प्रचार", "अर्थ", "वाक्य"],
            ),

            MarathiTopic(
                "proverbs",
                "म्हणी",
                "भाषा",
                "मराठी म्हणींचे अर्थ आणि त्यांचा योग्य उपयोग.",
                "easy",
                ["म्हण", "अर्थ"],
            ),

            MarathiTopic(
                "comprehension",
                "आकलन",
                "पठन",
                "गद्य किंवा पद्य उतारा वाचून प्रश्नांची उत्तरे देणे.",
                "medium",
                ["उतारा", "आकलन", "प्रश्नोत्तरे"],
            ),

            MarathiTopic(
                "translation",
                "भाषांतर",
                "भाषा",
                "मराठी आणि इतर भाषांमधील अर्थपूर्ण भाषांतराचा अभ्यास.",
                "medium",
                ["भाषांतर", "अनुवाद"],
            ),

            MarathiTopic(
                "essay",
                "निबंधलेखन",
                "लेखन",
                "दिलेल्या विषयावर सुसंगत आणि प्रभावी निबंध लिहिणे.",
                "medium",
                ["निबंध", "लेखन", "विचार"],
            ),

            MarathiTopic(
                "letter",
                "पत्रलेखन",
                "लेखन",
                "औपचारिक आणि अनौपचारिक पत्रलेखनाचा सराव.",
                "medium",
                ["पत्र", "औपचारिक", "अनौपचारिक"],
            ),

            MarathiTopic(
                "story",
                "कथालेखन",
                "लेखन",
                "दिलेल्या कल्पना किंवा मुद्द्यांवर आधारित कथा तयार करणे.",
                "medium",
                ["कथा", "कल्पनाविस्तार"],
            ),

            MarathiTopic(
                "summary",
                "सारांशलेखन",
                "लेखन",
                "मोठ्या उताऱ्यातील मुख्य आशय संक्षिप्त स्वरूपात मांडणे.",
                "hard",
                ["सारांश", "मुख्य आशय"],
            ),

            MarathiTopic(
                "report",
                "वृत्तांतलेखन",
                "लेखन",
                "घडलेल्या घटना किंवा कार्यक्रमाचा क्रमबद्ध वृत्तांत लिहिणे.",
                "medium",
                ["वृत्तांत", "कार्यक्रम", "घटना"],
            ),

            MarathiTopic(
                "dialogue",
                "संवादलेखन",
                "लेखन",
                "दोन किंवा अधिक व्यक्तींमधील संवाद प्रभावीपणे लिहिणे.",
                "medium",
                ["संवाद", "लेखन"],
            ),
        ]

        for topic in topics:
            self.register_topic(topic)

    def register_topic(
        self,
        topic: MarathiTopic,
    ) -> None:

        self.topics[
            self._normalize(topic.topic_id)
        ] = topic

    # ========================================================
    # TOPIC ACCESS
    # ========================================================

    def get_topic(
        self,
        topic: str,
    ) -> MarathiTopic | None:

        return self.topics.get(
            self._normalize(topic)
        )

    def get_topics(
        self,
        category: str | None = None,
    ) -> list[dict[str, Any]]:

        topics = list(self.topics.values())

        if category:

            category = category.casefold()

            topics = [
                topic
                for topic in topics
                if topic.category.casefold()
                == category
            ]

        return [
            topic.to_dict()
            for topic in topics
        ]

    def search_topics(
        self,
        query: str,
    ) -> list[dict[str, Any]]:

        query = self._normalize(query)

        results = []

        for topic in self.topics.values():

            searchable = self._normalize(
                " ".join(
                    [
                        topic.topic_id,
                        topic.name,
                        topic.category,
                        topic.description,
                        *topic.keywords,
                    ]
                )
            )

            if query in searchable:
                results.append(
                    topic.to_dict()
                )

        return results

    # ========================================================
    # QUESTION GENERATION
    # ========================================================

    def generate_question(
        self,
        topic: str,
        *,
        difficulty: str = "medium",
    ) -> MarathiQuestion:

        topic_id = self._normalize(topic)

        generators = {

            "grammar":
                self._grammar_question,

            "sandhi":
                self._sandhi_question,

            "samas":
                self._samas_question,

            "alankar":
                self._alankar_question,

            "kavita":
                self._kavita_question,

            "prose":
                self._prose_question,

            "literature":
                self._literature_question,

            "vocabulary":
                self._vocabulary_question,

            "idioms":
                self._idiom_question,

            "proverbs":
                self._proverb_question,

            "comprehension":
                self._comprehension_question,

            "translation":
                self._translation_question,

            "essay":
                self._essay_question,

            "letter":
                self._letter_question,

            "story":
                self._story_question,

            "summary":
                self._summary_question,

            "report":
                self._report_question,

            "dialogue":
                self._dialogue_question,
        }

        generator = generators.get(topic_id)

        if generator is None:

            return self._generic_question(
                topic_id,
                difficulty,
            )

        return generator(difficulty)

    def generate_questions(
        self,
        topic: str,
        count: int = 10,
        *,
        difficulty: str = "medium",
    ) -> list[MarathiQuestion]:

        if count <= 0:
            return []

        return [
            self.generate_question(
                topic,
                difficulty=difficulty,
            )
            for _ in range(count)
        ]

    # ========================================================
    # GRAMMAR
    # ========================================================

    def _grammar_question(
        self,
        difficulty: str,
    ) -> MarathiQuestion:

        questions = [

            (
                "'राम शाळेत जातो.' या वाक्यातील कर्ता कोणता?",
                "राम",
                "'राम' हा क्रिया करणारा असल्यामुळे कर्ता आहे.",
                "क्रिया कोण करतो हे शोधा.",
            ),

            (
                "'सीमा पुस्तक वाचते.' या वाक्यातील क्रियापद ओळखा.",
                "वाचते",
                "'वाचते' हा क्रिया दर्शवणारा शब्द आहे.",
                "वाक्यातील कृती दर्शवणारा शब्द शोधा.",
            ),

            (
                "'सुंदर फूल' या शब्दसमूहातील विशेषण कोणते?",
                "सुंदर",
                "'सुंदर' हा 'फूल' या नामाची विशेषता सांगतो.",
                "नामाची विशेषता सांगणारा शब्द शोधा.",
            ),
        ]

        return self._choice_question(
            "grammar",
            "व्याकरण",
            questions,
            difficulty,
            marks=1,
        )

    # ========================================================
    # SANDHI
    # ========================================================

    def _sandhi_question(
        self,
        difficulty: str,
    ) -> MarathiQuestion:

        questions = [

            (
                "'विद्यालय' या शब्दाची संधी सोडवा.",
                "विद्या + आलय",
                "विद्यालय = विद्या + आलय.",
                "शब्दाचे दोन मूळ भाग शोधा.",
            ),

            (
                "'देवालय' या शब्दाची संधी सोडवा.",
                "देव + आलय",
                "देवालय = देव + आलय.",
                "देव आणि आलय हे दोन शब्द लक्षात घ्या.",
            ),

            (
                "'हिमालय' या शब्दाची संधी सोडवा.",
                "हिम + आलय",
                "हिमालय = हिम + आलय.",
                "हिम आणि आलय वेगळे करा.",
            ),
        ]

        return self._choice_question(
            "sandhi",
            "व्याकरण",
            questions,
            difficulty,
            marks=2,
        )

    # ========================================================
    # SAMAS
    # ========================================================

    def _samas_question(
        self,
        difficulty: str,
    ) -> MarathiQuestion:

        questions = [

            (
                "'राजपुत्र' या शब्दाचा विग्रह करा.",
                "राजाचा पुत्र",
                "राजपुत्र = राजाचा पुत्र.",
                "दोन्ही शब्दांमधील संबंध ओळखा.",
            ),

            (
                "'नीलकमल' या शब्दाचा विग्रह करा.",
                "निळे कमळ",
                "नीलकमल = निळे कमळ.",
                "विशेषण आणि नाम ओळखा.",
            ),

            (
                "'आईवडील' हा कोणता समास आहे?",
                "द्वंद्व समास",
                "आई आणि वडील या दोन्ही पदांना समान महत्त्व आहे.",
                "दोन्ही पदे समान महत्त्वाची आहेत का ते पहा.",
            ),
        ]

        return self._choice_question(
            "samas",
            "व्याकरण",
            questions,
            difficulty,
            marks=2,
        )

    # ========================================================
    # ALANKAR
    # ========================================================

    def _alankar_question(
        self,
        difficulty: str,
    ) -> MarathiQuestion:

        questions = [

            (
                "'तिचे मुख चंद्रासारखे सुंदर आहे.' "
                "या वाक्यातील अलंकार कोणता?",
                "उपमा अलंकार",
                "'सारखे' या शब्दाने तुलना केली असल्यामुळे उपमा अलंकार आहे.",
                "तुलना दर्शवणारा शब्द शोधा.",
            ),

            (
                "'तो रणांगणातील सिंह आहे.' "
                "या वाक्यातील अलंकार कोणता?",
                "रूपक अलंकार",
                "व्यक्तीला थेट सिंह म्हटले आहे, त्यामुळे रूपक अलंकार आहे.",
                "थेट एकरूपता दाखवली आहे का ते पहा.",
            ),

            (
                "'चंचल चपळ चिमणी' या शब्दसमूहात कोणता अलंकार आहे?",
                "अनुप्रास अलंकार",
                "समान ध्वनीची पुनरावृत्ती झाल्यामुळे अनुप्रास अलंकार आहे.",
                "समान अक्षर/ध्वनीची पुनरावृत्ती पहा.",
            ),
        ]

        return self._choice_question(
            "alankar",
            "व्याकरण",
            questions,
            difficulty,
            marks=2,
        )

    # ========================================================
    # KAVITA
    # ========================================================

    def _kavita_question(
        self,
        difficulty: str,
    ) -> MarathiQuestion:

        questions = [

            (
                "कवितेचा 'भावार्थ' म्हणजे काय?",
                "कवितेत व्यक्त झालेला मुख्य भाव आणि आशय स्वतःच्या शब्दांत मांडणे.",
                "भावार्थात कवितेचा मुख्य आशय स्पष्टपणे मांडला जातो.",
                "कवितेचा मुख्य संदेश स्वतःच्या शब्दांत लिहा.",
            ),

            (
                "कवितेचे रसग्रहण करताना कोणत्या गोष्टींचा विचार केला जातो?",
                "आशय, भाव, भाषा, काव्यसौंदर्य आणि संदेश यांचा विचार केला जातो.",
                "रसग्रहण म्हणजे कवितेतील साहित्यिक सौंदर्य समजून घेणे.",
                "कवितेतील आशय आणि सौंदर्य दोन्ही पहा.",
            ),
        ]

        return self._choice_question(
            "kavita",
            "साहित्य",
            questions,
            difficulty,
            marks=3,
        )

    # ========================================================
    # PROSE
    # ========================================================

    def _prose_question(
        self,
        difficulty: str,
    ) -> MarathiQuestion:

        questions = [

            (
                "गद्यलेखन म्हणजे काय?",
                "छंद किंवा वृत्ताच्या बंधनाशिवाय वाक्यरचनेत केलेले लेखन म्हणजे गद्यलेखन.",
                "गद्याची रचना सामान्य वाक्यांच्या स्वरूपात असते.",
                "कवितेच्या तुलनेत गद्याची रचना कशी असते ते आठवा.",
            ),

            (
                "गद्य उताऱ्याचा मुख्य आशय कसा शोधावा?",
                "उताऱ्यातील मुख्य विचार, घटना आणि लेखकाचा संदेश ओळखून.",
                "मुख्य आशय म्हणजे संपूर्ण उताऱ्याचा केंद्रबिंदू.",
                "प्रत्येक परिच्छेदातील मुख्य विचार शोधा.",
            ),
        ]

        return self._choice_question(
            "prose",
            "साहित्य",
            questions,
            difficulty,
            marks=3,
        )

    # ========================================================
    # LITERATURE
    # ========================================================

    def _literature_question(
        self,
        difficulty: str,
    ) -> MarathiQuestion:

        questions = [

            (
                "मराठी साहित्याचे प्रमुख प्रकार कोणते?",
                "कविता, कथा, कादंबरी, नाटक, निबंध, चरित्र आणि आत्मचरित्र इत्यादी.",
                "मराठी साहित्य विविध गद्य आणि पद्य प्रकारांनी समृद्ध आहे.",
                "गद्य आणि पद्य दोन्ही प्रकार आठवा.",
            ),

            (
                "साहित्याचा समाजावर काय परिणाम होतो?",
                "साहित्य विचार, भावना, संस्कृती आणि सामाजिक जाणीवा विकसित करण्यास मदत करते.",
                "साहित्य समाजाचे प्रतिबिंब दाखवते आणि विचारांना दिशा देऊ शकते.",
                "विचार आणि संस्कृती यांचा संबंध पहा.",
            ),
        ]

        return self._choice_question(
            "literature",
            "साहित्य",
            questions,
            difficulty,
            marks=3,
        )

    # ========================================================
    # VOCABULARY
    # ========================================================

    def _vocabulary_question(
        self,
        difficulty: str,
    ) -> MarathiQuestion:

        questions = [

            (
                "'सूर्य' या शब्दाचा समानार्थी शब्द कोणता?",
                "रवि",
                "सूर्यचे समानार्थी शब्द रवि, भास्कर, दिनकर इत्यादी आहेत.",
                "सूर्यासाठी वापरला जाणारा काव्यात्मक शब्द आठवा.",
            ),

            (
                "'आनंद' या शब्दाचा विरुद्धार्थी शब्द कोणता?",
                "दुःख",
                "आनंदचा विरुद्धार्थी शब्द दुःख आहे.",
                "आनंदाच्या विरुद्ध भावना कोणती?",
            ),

            (
                "'जल' या शब्दाचा अर्थ काय?",
                "पाणी",
                "जल म्हणजे पाणी.",
                "जल हा पाण्यासाठी वापरला जाणारा शब्द आहे.",
            ),

            (
                "'गृह' या शब्दाचा अर्थ काय?",
                "घर",
                "गृह म्हणजे घर.",
                "गृह आणि घर हे समान अर्थाचे शब्द आहेत.",
            ),
        ]

        return self._choice_question(
            "vocabulary",
            "भाषा",
            questions,
            difficulty,
            marks=1,
        )

    # ========================================================
    # IDIOMS
    # ========================================================

    def _idiom_question(
        self,
        difficulty: str,
    ) -> MarathiQuestion:

        questions = [

            (
                "'डोळ्यात धूळ फेकणे' या वाक्प्रचाराचा अर्थ काय?",
                "फसवणे",
                "डोळ्यात धूळ फेकणे म्हणजे एखाद्याला फसवणे.",
                "एखाद्याला सत्य न कळू देणे असा अर्थ घ्या.",
            ),

            (
                "'नाक खुपसणे' या वाक्प्रचाराचा अर्थ काय?",
                "विनाकारण हस्तक्षेप करणे",
                "नाक खुपसणे म्हणजे दुसऱ्याच्या कामात विनाकारण हस्तक्षेप करणे.",
                "दुसऱ्याच्या कामात हस्तक्षेप करण्याचा अर्थ आठवा.",
            ),

            (
                "'हात टेकणे' या वाक्प्रचाराचा अर्थ काय?",
                "हार मानणे",
                "हात टेकणे म्हणजे प्रयत्न सोडून देणे किंवा हार मानणे.",
                "पराभवानंतर काय केले जाते?",
            ),
        ]

        return self._choice_question(
            "idioms",
            "भाषा",
            questions,
            difficulty,
            marks=2,
        )

    # ========================================================
    # PROVERBS
    # ========================================================

    def _proverb_question(
        self,
        difficulty: str,
    ) -> MarathiQuestion:

        questions = [

            (
                "'थेंबे थेंबे तळे साचे' या म्हणीचा अर्थ काय?",
                "थोडे थोडे जमा केल्यास मोठा संचय होतो.",
                "लहान प्रयत्न आणि बचत यांमुळे मोठा परिणाम निर्माण होतो.",
                "थेंब एकत्र आल्यावर काय तयार होते?",
            ),

            (
                "'अति तिथे माती' या म्हणीचा अर्थ काय?",
                "कोणत्याही गोष्टीचा अतिरेक वाईट असतो.",
                "अतिरेकामुळे चांगली गोष्टही नुकसानकारक होऊ शकते.",
                "अतिरेकाचा परिणाम काय होतो याचा विचार करा.",
            ),

            (
                "'जशी करणी तशी भरणी' या म्हणीचा अर्थ काय?",
                "जसे कर्म कराल तसेच त्याचे फळ मिळेल.",
                "कर्म आणि परिणाम यांचा संबंध या म्हणीत सांगितला आहे.",
                "आपल्या कृतींचे परिणाम काय असतात?",
            ),
        ]

        return self._choice_question(
            "proverbs",
            "भाषा",
            questions,
            difficulty,
            marks=2,
        )

    # ========================================================
    # COMPREHENSION
    # ========================================================

    def _comprehension_question(
        self,
        difficulty: str,
    ) -> MarathiQuestion:

        passage = (
            "एक शेतकरी दररोज पहाटे शेतात जात असे. "
            "तो मेहनतीने जमीन कसत असे. "
            "त्याला आपल्या कष्टावर पूर्ण विश्वास होता. "
            "पावसाची वाट पाहत बसण्याऐवजी तो उपलब्ध "
            "पाण्याचा योग्य उपयोग करत असे."
        )

        return self._question(
            topic="comprehension",
            category="पठन",
            question=(
                f"उतारा वाचा आणि उत्तर द्या.\n\n"
                f"{passage}\n\n"
                f"शेतकरी उपलब्ध पाण्याचा कसा उपयोग करत असे?"
            ),
            answer="तो उपलब्ध पाण्याचा योग्य उपयोग करत असे.",
            difficulty=difficulty,
            explanation=(
                "उताऱ्यात स्पष्टपणे सांगितले आहे की "
                "शेतकरी उपलब्ध पाण्याचा योग्य उपयोग करत असे."
            ),
            hint="उताऱ्याच्या शेवटच्या वाक्याकडे पहा.",
            marks=2,
        )

    # ========================================================
    # TRANSLATION
    # ========================================================

    def _translation_question(
        self,
        difficulty: str,
    ) -> MarathiQuestion:

        questions = [

            (
                "'राम शाळेत जातो.' या वाक्याचे इंग्रजी भाषांतर करा.",
                "Ram goes to school.",
                "राम = Ram, शाळेत = to school, जातो = goes.",
                "कर्ता आणि क्रियापद ओळखा.",
            ),

            (
                "'मुले मैदानात खेळतात.' या वाक्याचे इंग्रजी भाषांतर करा.",
                "The children play in the मैदान.",
                "मुले = children, मैदानात = in the मैदान, खेळतात = play.",
                "बहुवचन कर्त्याकडे लक्ष द्या.",
            ),

            (
                "'सीमा पुस्तक वाचते.' या वाक्याचे इंग्रजी भाषांतर करा.",
                "Seema reads a book.",
                "सीमा = Seema, पुस्तक = book, वाचते = reads.",
                "क्रियापदाचा योग्य काळ वापरा.",
            ),
        ]

        return self._choice_question(
            "translation",
            "भाषा",
            questions,
            difficulty,
            marks=2,
        )

    # ========================================================
    # ESSAY
    # ========================================================

    def _essay_question(
        self,
        difficulty: str,
    ) -> MarathiQuestion:

        topics = [
            "माझे आवडते पुस्तक",
            "पर्यावरणाचे महत्त्व",
            "माझा आवडता खेळ",
            "विद्यार्थी जीवन",
            "वेळेचे महत्त्व",
            "वाचनाचे महत्त्व",
        ]

        topic = self.random.choice(
            topics
        )

        return self._question(
            topic="essay",
            category="लेखन",
            question=(
                f"'{topic}' या विषयावर "
                f"निबंध लिहा."
            ),
            answer=(
                "उत्तरामध्ये प्रस्तावना, मुख्य विचार, "
                "उदाहरणे आणि योग्य निष्कर्ष असावा."
            ),
            difficulty=difficulty,
            explanation=(
                "निबंध सुसंगत परिच्छेदांमध्ये लिहावा. "
                "विषयाची प्रस्तावना करून मुख्य मुद्दे "
                "स्पष्ट करावेत आणि शेवटी निष्कर्ष द्यावा."
            ),
            hint=(
                "प्रस्तावना → मुख्य मुद्दे → उदाहरणे → निष्कर्ष"
            ),
            marks=5,
        )

    # ========================================================
    # LETTER
    # ========================================================

    def _letter_question(
        self,
        difficulty: str,
    ) -> MarathiQuestion:

        prompts = [

            (
                "तुमच्या मित्राला वाढदिवसाच्या शुभेच्छा देणारे "
                "अनौपचारिक पत्र लिहा."
            ),

            (
                "शाळेत स्वच्छता अभियान आयोजित करण्यासाठी "
                "मुख्याध्यापकांना विनंतीपत्र लिहा."
            ),

            (
                "तुमच्या मित्राला परीक्षेच्या तयारीबद्दल "
                "पत्र लिहा."
            ),
        ]

        return self._question(
            topic="letter",
            category="लेखन",
            question=self.random.choice(
                prompts
            ),
            answer=(
                "पत्रामध्ये योग्य मायना, विषयानुसार "
                "मजकूर आणि शेवटचा योग्य नमुना असावा."
            ),
            difficulty=difficulty,
            explanation=(
                "पत्राचा प्रकार ओळखून योग्य रचना वापरा. "
                "अनौपचारिक पत्रात आत्मीय भाषा आणि "
                "औपचारिक पत्रात नम्र व औपचारिक भाषा वापरा."
            ),
            hint=(
                "पत्ता → दिनांक → मायना → मजकूर → शेवट"
            ),
            marks=5,
        )

    # ========================================================
    # STORY
    # ========================================================

    def _story_question(
        self,
        difficulty: str,
    ) -> MarathiQuestion:

        prompts = [

            "खालील मुद्द्यांच्या आधारे कथा लिहा: "
            "एक विद्यार्थी — रस्त्यावर सापडलेले पाकीट — "
            "पाकिटात पैसे — मालकाचा शोध — प्रामाणिकपणा.",

            "कल्पनाविस्तार करा: "
            "एक छोटा निर्णय आणि त्याने बदललेले संपूर्ण जीवन.",

            "खालील शेवटावर आधारित कथा लिहा: "
            "'त्या दिवसापासून त्याने कधीही खोटे बोलले नाही.'",
        ]

        return self._question(
            topic="story",
            category="लेखन",
            question=self.random.choice(
                prompts
            ),
            answer=(
                "कथेची सुरुवात, घटना, उत्कंठा, "
                "परिणाम आणि योग्य शेवट असावा."
            ),
            difficulty=difficulty,
            explanation=(
                "कथा क्रमबद्ध असावी आणि तिच्यात "
                "योग्य शीर्षक व बोध असू शकतो."
            ),
            hint=(
                "सुरुवात → घटना → संघर्ष → परिणाम → शेवट"
            ),
            marks=5,
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    def _summary_question(
        self,
        difficulty: str,
    ) -> MarathiQuestion:

        passage = (
            "वाचन ही ज्ञान मिळवण्याची अत्यंत प्रभावी सवय आहे. "
            "नियमित वाचनामुळे शब्दसंपत्ती वाढते, विचारशक्ती "
            "विकसित होते आणि नवीन विषय समजण्याची क्षमता "
            "वाढते. चांगली पुस्तके माणसाच्या व्यक्तिमत्त्वावर "
            "सकारात्मक परिणाम करतात."
        )

        return self._question(
            topic="summary",
            category="लेखन",
            question=(
                f"खालील उताऱ्याचा संक्षिप्त सारांश लिहा:\n\n"
                f"{passage}"
            ),
            answer=(
                "नियमित वाचनामुळे ज्ञान, शब्दसंपत्ती "
                "आणि विचारशक्ती वाढते तसेच व्यक्तिमत्त्व "
                "विकसित होण्यास मदत होते."
            ),
            difficulty=difficulty,
            explanation=(
                "सारांशात मूळ उताऱ्याचा मुख्य आशय "
                "संक्षिप्तपणे मांडला पाहिजे."
            ),
            hint="वाचनाचे मुख्य फायदे शोधा.",
            marks=5,
        )

    # ========================================================
    # REPORT
    # ========================================================

    def _report_question(
        self,
        difficulty: str,
    ) -> MarathiQuestion:

        return self._question(
            topic="report",
            category="लेखन",
            question=(
                "तुमच्या शाळेत झालेल्या क्रीडा दिनाचा "
                "वृत्तांत लिहा."
            ),
            answer=(
                "वृत्तांतात कार्यक्रमाची तारीख, ठिकाण, "
                "प्रमुख घटना, सहभागी आणि निष्कर्ष यांचा "
                "समावेश असावा."
            ),
            difficulty=difficulty,
            explanation=(
                "वृत्तांत वस्तुनिष्ठ, क्रमबद्ध आणि स्पष्ट असावा."
            ),
            hint=(
                "कधी → कुठे → काय झाले → कोण सहभागी झाले → परिणाम"
            ),
            marks=5,
        )

    # ========================================================
    # DIALOGUE
    # ========================================================

    def _dialogue_question(
        self,
        difficulty: str,
    ) -> MarathiQuestion:

        prompts = [

            "दोन मित्रांमधील 'परीक्षेची तयारी' या विषयावरील संवाद लिहा.",

            "विद्यार्थी आणि शिक्षक यांच्यातील "
            "'वेळेचे नियोजन' या विषयावरील संवाद लिहा.",

            "दोन मित्रांमधील 'मोबाईलचा योग्य वापर' "
            "या विषयावरील संवाद लिहा.",
        ]

        return self._question(
            topic="dialogue",
            category="लेखन",
            question=self.random.choice(
                prompts
            ),
            answer=(
                "संवाद नैसर्गिक, विषयाला धरून आणि "
                "दोन्ही व्यक्तींच्या योग्य प्रतिसादांसह असावा."
            ),
            difficulty=difficulty,
            explanation=(
                "संवादात प्रश्न-उत्तरांची नैसर्गिक देवाणघेवाण "
                "असावी. अनावश्यक वर्णन टाळावे."
            ),
            hint=(
                "व्यक्ती १ → प्रतिक्रिया → व्यक्ती २ → प्रतिक्रिया"
            ),
            marks=5,
        )

    # ========================================================
    # QUESTION HELPERS
    # ========================================================

    def _choice_question(
        self,
        topic: str,
        category: str,
        questions: list[tuple[str, str, str, str]],
        difficulty: str,
        *,
        marks: int,
    ) -> MarathiQuestion:

        selected = self.random.choice(
            questions
        )

        return self._question(
            topic=topic,
            category=category,
            question=selected[0],
            answer=selected[1],
            difficulty=difficulty,
            explanation=selected[2],
            hint=selected[3],
            marks=marks,
        )

    def _question(
        self,
        *,
        topic: str,
        category: str,
        question: str,
        answer: Any,
        difficulty: str,
        explanation: str,
        hint: str,
        marks: int,
        options: list[str] | None = None,
    ) -> MarathiQuestion:

        self.question_counter += 1

        return MarathiQuestion(
            question_id=(
                f"MAR-{self.question_counter:06d}"
            ),
            topic=topic,
            category=category,
            question=question,
            answer=answer,
            options=options or [],
            difficulty=difficulty,
            explanation=explanation,
            hint=hint,
            marks=marks,
        )

    def _generic_question(
        self,
        topic: str,
        difficulty: str,
    ) -> MarathiQuestion:

        data = self.get_topic(topic)

        if data is None:

            return self._question(
                topic=topic,
                category="मराठी",
                question=(
                    f"'{topic}' या विषयाची "
                    f"थोडक्यात माहिती लिहा."
                ),
                answer="",
                difficulty=difficulty,
                explanation=(
                    "प्रथम विषयाची व्याख्या, "
                    "मुख्य वैशिष्ट्ये आणि उदाहरणे लिहा."
                ),
                hint="विषयाची व्याख्या लिहून सुरुवात करा.",
                marks=3,
            )

        return self._question(
            topic=topic,
            category=data.category,
            question=(
                f"'{data.name}' या विषयाचे "
                f"स्पष्टीकरण करा."
            ),
            answer=data.description,
            difficulty=difficulty,
            explanation=data.description,
            hint="प्रथम व्याख्या लिहा.",
            marks=3,
        )

    # ========================================================
    # ANSWER CHECKING
    # ========================================================

    def check_answer(
        self,
        question: MarathiQuestion,
        user_answer: str,
    ) -> dict[str, Any]:

        actual = self._normalize(
            user_answer
        )

        expected = self._normalize(
            question.answer
        )

        if not actual:

            return {
                "correct": False,
                "partially_correct": False,
                "score": 0,
                "max_score": question.marks,
                "accuracy": 0.0,
                "question_id": question.question_id,
                "correct_answer": question.answer,
                "feedback": "उत्तर रिकामे आहे.",
                "hint": question.hint,
                "explanation": question.explanation,
            }

        if actual == expected:

            return {
                "correct": True,
                "partially_correct": False,
                "score": question.marks,
                "max_score": question.marks,
                "accuracy": 100.0,
                "question_id": question.question_id,
                "correct_answer": question.answer,
                "user_answer": user_answer,
                "feedback": "उत्तम! उत्तर पूर्णपणे बरोबर आहे.",
                "explanation": question.explanation,
            }

        similarity = self._similarity(
            actual,
            expected,
        )

        if similarity >= 0.75:

            score = max(
                1,
                int(
                    question.marks
                    * similarity
                ),
            )

            return {
                "correct": True,
                "partially_correct": True,
                "score": score,
                "max_score": question.marks,
                "accuracy": round(
                    similarity * 100,
                    2,
                ),
                "question_id": question.question_id,
                "correct_answer": question.answer,
                "user_answer": user_answer,
                "feedback": (
                    "मुख्य उत्तर बरोबर आहे; "
                    "थोडी अधिक अचूकता आवश्यक आहे."
                ),
                "explanation": question.explanation,
            }

        if similarity >= 0.40:

            score = int(
                question.marks
                * similarity
            )

            return {
                "correct": False,
                "partially_correct": True,
                "score": score,
                "max_score": question.marks,
                "accuracy": round(
                    similarity * 100,
                    2,
                ),
                "question_id": question.question_id,
                "correct_answer": question.answer,
                "user_answer": user_answer,
                "feedback": (
                    "उत्तरामध्ये काही योग्य मुद्दे आहेत, "
                    "पण पूर्ण उत्तर अपेक्षित आहे."
                ),
                "hint": question.hint,
                "explanation": question.explanation,
            }

        return {
            "correct": False,
            "partially_correct": False,
            "score": 0,
            "max_score": question.marks,
            "accuracy": round(
                similarity * 100,
                2,
            ),
            "question_id": question.question_id,
            "correct_answer": question.answer,
            "user_answer": user_answer,
            "feedback": (
                "उत्तर चुकीचे आहे. "
                "स्पष्टीकरण पाहून पुन्हा प्रयत्न करा."
            ),
            "hint": question.hint,
            "explanation": question.explanation,
        }

    # ========================================================
    # TEXT UTILITIES
    # ========================================================

    @staticmethod
    def _normalize(
        text: Any,
    ) -> str:

        if text is None:
            return ""

        value = str(text).casefold()

        value = re.sub(
            r"[^\w\u0900-\u097F\s]",
            " ",
            value,
        )

        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        return value.strip()

    @staticmethod
    def _similarity(
        actual: str,
        expected: str,
    ) -> float:

        if not actual or not expected:
            return 0.0

        actual_words = set(
            actual.split()
        )

        expected_words = set(
            expected.split()
        )

        if not expected_words:
            return 0.0

        common = (
            actual_words
            & expected_words
        )

        return min(
            1.0,
            len(common)
            / len(expected_words),
        )

    # ========================================================
    # PRACTICE
    # ========================================================

    def practice_grammar(
        self,
        count: int = 10,
        *,
        difficulty: str = "medium",
    ) -> list[MarathiQuestion]:

        topics = [
            "grammar",
            "sandhi",
            "samas",
            "alankar",
        ]

        return [
            self.generate_question(
                self.random.choice(topics),
                difficulty=difficulty,
            )
            for _ in range(max(0, count))
        ]

    def practice_language(
        self,
        count: int = 10,
        *,
        difficulty: str = "medium",
    ) -> list[MarathiQuestion]:

        topics = [
            "vocabulary",
            "idioms",
            "proverbs",
            "translation",
        ]

        return [
            self.generate_question(
                self.random.choice(topics),
                difficulty=difficulty,
            )
            for _ in range(max(0, count))
        ]

    def practice_writing(
        self,
        count: int = 10,
        *,
        difficulty: str = "medium",
    ) -> list[MarathiQuestion]:

        topics = [
            "essay",
            "letter",
            "story",
            "summary",
            "report",
            "dialogue",
        ]

        return [
            self.generate_question(
                self.random.choice(topics),
                difficulty=difficulty,
            )
            for _ in range(max(0, count))
        ]

    # ========================================================
    # RANDOM QUESTIONS
    # ========================================================

    def random_topic(
        self,
        *,
        category: str | None = None,
    ) -> MarathiTopic | None:

        topics = list(
            self.topics.values()
        )

        if category:

            category = category.casefold()

            topics = [
                topic
                for topic in topics
                if topic.category.casefold()
                == category
            ]

        if not topics:
            return None

        return self.random.choice(
            topics
        )

    def generate_random_question(
        self,
        *,
        category: str | None = None,
        difficulty: str = "medium",
    ) -> MarathiQuestion:

        topic = self.random_topic(
            category=category
        )

        if topic is None:

            return self._generic_question(
                "marathi",
                difficulty,
            )

        return self.generate_question(
            topic.topic_id,
            difficulty=difficulty,
        )

    # ========================================================
    # REVISION SET
    # ========================================================

    def create_revision_set(
        self,
        *,
        category: str | None = None,
        count: int = 10,
        difficulty: str = "medium",
    ) -> list[MarathiQuestion]:

        if count <= 0:
            return []

        topics = list(
            self.topics.values()
        )

        if category:

            category = category.casefold()

            topics = [
                topic
                for topic in topics
                if topic.category.casefold()
                == category
            ]

        self.random.shuffle(
            topics
        )

        questions: list[
            MarathiQuestion
        ] = []

        for topic in topics:

            if len(questions) >= count:
                break

            questions.append(
                self.generate_question(
                    topic.topic_id,
                    difficulty=difficulty,
                )
            )

        while len(questions) < count:

            topic = self.random_topic(
                category=category
            )

            if topic is None:
                break

            questions.append(
                self.generate_question(
                    topic.topic_id,
                    difficulty=difficulty,
                )
            )

        return questions

    # ========================================================
    # DIFFICULTY
    # ========================================================

    @staticmethod
    def difficulty_marks(
        difficulty: str,
    ) -> int:

        mapping = {
            "easy": 1,
            "medium": 3,
            "hard": 5,
        }

        return mapping.get(
            difficulty.casefold(),
            3,
        )

    def classify_question_difficulty(
        self,
        question: str,
    ) -> str:

        text = self._normalize(
            question
        )

        hard_terms = [
            "स्पष्टीकरण",
            "विश्लेषण",
            "तुलना",
            "भेद",
            "भावार्थ",
            "रसग्रहण",
            "संधी",
            "समास",
            "सारांश",
            "निबंध",
        ]

        easy_terms = [
            "काय",
            "कोण",
            "कुठे",
            "कधी",
            "अर्थ",
            "नाव",
            "ओळखा",
        ]

        if any(
            term in text
            for term in hard_terms
        ):
            return "hard"

        if any(
            term in text
            for term in easy_terms
        ):
            return "easy"

        return "medium"

    # ========================================================
    # DICTIONARY
    # ========================================================

    def get_basic_dictionary(
        self,
    ) -> dict[str, str]:

        return {
            "सूर्य": "रवि",
            "चंद्र": "शशी",
            "पाणी": "जल",
            "घर": "गृह",
            "शाळा": "विद्यालय",
            "मित्र": "सखा",
            "आई": "माता",
            "वडील": "पिता",
            "मुलगा": "पुत्र",
            "मुलगी": "कन्या",
            "फूल": "पुष्प",
            "झाड": "वृक्ष",
            "आकाश": "नभ",
            "पृथ्वी": "धरती",
            "आनंद": "सुख",
            "दुःख": "शोक",
            "प्रकाश": "तेज",
            "रात्र": "रजनी",
            "दिवस": "दिन",
            "समुद्र": "सागर",
        }

    def word_meaning(
        self,
        word: str,
    ) -> str | None:

        return self.get_basic_dictionary().get(
            word.strip()
        )

    # ========================================================
    # TOPIC RECOMMENDATIONS
    # ========================================================

    def recommend_topics(
        self,
        *,
        weak_topics: Iterable[str] | None = None,
        category: str | None = None,
        count: int = 5,
    ) -> list[str]:

        if weak_topics:

            return list(
                weak_topics
            )[:count]

        topics = list(
            self.topics.values()
        )

        if category:

            category = category.casefold()

            topics = [
                topic
                for topic in topics
                if topic.category.casefold()
                == category
            ]

        self.random.shuffle(
            topics
        )

        return [
            topic.topic_id
            for topic in topics[:count]
        ]


# ============================================================
# FACTORY
# ============================================================


def create_marathi_engine(
    *,
    seed: int | None = None,
) -> MarathiEngine:

    return MarathiEngine(
        seed=seed
    )


# ============================================================
# EXPORTS
# ============================================================


__all__ = [
    "MarathiTopic",
    "MarathiQuestion",
    "MarathiEngine",
    "create_marathi_engine",
]


