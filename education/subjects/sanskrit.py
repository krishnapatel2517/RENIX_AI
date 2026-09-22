"""
RENIX Education — Sanskrit Subject Engine

File:
    RENIX/education/subjects/sanskrit.py

Purpose:
    Sanskrit learning, practice, revision, vocabulary,
    grammar, translation, question generation and
    answer checking for the RENIX education system.
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
class SanskritTopic:
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
class SanskritQuestion:
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
# SANSKRIT ENGINE
# ============================================================


class SanskritEngine:

    def __init__(
        self,
        *,
        seed: int | None = None,
    ) -> None:

        self.random = random.Random(seed)

        self.topics: dict[str, SanskritTopic] = {}

        self.question_counter = 0

        self._register_topics()

    # ========================================================
    # TOPICS
    # ========================================================

    def _register_topics(self) -> None:

        topics = [

            SanskritTopic(
                "sandhi",
                "सन्धिः",
                "व्याकरण",
                "वर्णों के मेल से होने वाले ध्वनि परिवर्तन का अध्ययन।",
                "hard",
                ["स्वरसन्धि", "व्यञ्जनसन्धि", "विसर्गसन्धि"],
            ),

            SanskritTopic(
                "samas",
                "समासः",
                "व्याकरण",
                "दो या दो से अधिक पदों के संक्षिप्त रूप से बनने वाली समस्त पद-रचना।",
                "hard",
                ["द्वन्द्व", "तत्पुरुष", "बहुव्रीहि", "अव्ययीभाव"],
            ),

            SanskritTopic(
                "karaka",
                "कारक",
                "व्याकरण",
                "क्रिया के साथ संज्ञा या सर्वनाम के संबंध का अध्ययन।",
                "medium",
                ["कर्ता", "कर्म", "करण", "सम्प्रदान", "अपादान", "अधिकरण"],
            ),

            SanskritTopic(
                "vibhakti",
                "विभक्तिः",
                "व्याकरण",
                "संज्ञा और सर्वनाम के विभिन्न रूपों का अध्ययन।",
                "medium",
                ["प्रथमा", "द्वितीया", "तृतीया", "चतुर्थी", "पञ्चमी", "षष्ठी", "सप्तमी"],
            ),

            SanskritTopic(
                "dhatu",
                "धातुरूप",
                "व्याकरण",
                "धातुओं के विभिन्न लकारों में रूपों का अध्ययन।",
                "hard",
                ["लट्", "लङ्", "लृट्", "लोट्", "विधिलिङ्"],
            ),

            SanskritTopic(
                "shabdarup",
                "शब्दरूप",
                "व्याकरण",
                "संज्ञा तथा सर्वनाम के रूपों का अध्ययन।",
                "hard",
                ["राम", "फल", "लता", "नदी", "मुनि"],
            ),

            SanskritTopic(
                "pratyaya",
                "प्रत्यय",
                "व्याकरण",
                "शब्द निर्माण में प्रयुक्त प्रत्ययों का अध्ययन।",
                "hard",
                ["कृत्", "तद्धित", "प्रत्यय"],
            ),

            SanskritTopic(
                "avyaya",
                "अव्यय",
                "व्याकरण",
                "जिन शब्दों के रूप में परिवर्तन नहीं होता उनका अध्ययन।",
                "easy",
                ["च", "अपि", "अत्र", "तत्र", "कुत्र"],
            ),

            SanskritTopic(
                "translation",
                "अनुवाद",
                "भाषा",
                "संस्कृत से हिन्दी/English तथा हिन्दी से संस्कृत अनुवाद।",
                "medium",
                ["संस्कृत", "हिन्दी", "English"],
            ),

            SanskritTopic(
                "vocabulary",
                "शब्दज्ञान",
                "भाषा",
                "संस्कृत शब्दों के अर्थ, पर्याय और विलोम का अभ्यास।",
                "easy",
                ["शब्दार्थ", "पर्यायवाची", "विलोम"],
            ),

            SanskritTopic(
                "comprehension",
                "अपठित-अवबोधन",
                "पठन",
                "अपठित गद्यांश को समझकर प्रश्नों के उत्तर देना।",
                "medium",
                ["गद्यांश", "प्रश्न", "उत्तर"],
            ),

            SanskritTopic(
                "sentence",
                "वाक्यरचना",
                "भाषा",
                "संस्कृत वाक्यों का निर्माण और सुधार।",
                "medium",
                ["वाक्य", "रचना", "शुद्धि"],
            ),

            SanskritTopic(
                "subhashita",
                "सुभाषित",
                "साहित्य",
                "नीतिपरक संस्कृत श्लोकों और उनके अर्थ का अध्ययन।",
                "medium",
                ["श्लोक", "नीति", "अर्थ"],
            ),

            SanskritTopic(
                "literature",
                "संस्कृत साहित्य",
                "साहित्य",
                "संस्कृत गद्य, पद्य, कथाओं और साहित्यिक विषयों का अध्ययन।",
                "medium",
                ["गद्य", "पद्य", "कथा", "श्लोक"],
            ),
        ]

        for topic in topics:
            self.register_topic(topic)

    def register_topic(
        self,
        topic: SanskritTopic,
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
    ) -> SanskritTopic | None:

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
    ) -> SanskritQuestion:

        topic_id = self._normalize(topic)

        generators = {

            "sandhi":
                self._sandhi_question,

            "samas":
                self._samas_question,

            "karaka":
                self._karaka_question,

            "vibhakti":
                self._vibhakti_question,

            "dhatu":
                self._dhatu_question,

            "shabdarup":
                self._shabdarup_question,

            "pratyaya":
                self._pratyaya_question,

            "avyaya":
                self._avyaya_question,

            "translation":
                self._translation_question,

            "vocabulary":
                self._vocabulary_question,

            "comprehension":
                self._comprehension_question,

            "sentence":
                self._sentence_question,

            "subhashita":
                self._subhashita_question,

            "literature":
                self._literature_question,
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
    ) -> list[SanskritQuestion]:

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
    # SANDHI
    # ========================================================

    def _sandhi_question(
        self,
        difficulty: str,
    ) -> SanskritQuestion:

        questions = [

            (
                "विद्यालयः इत्यस्य सन्धिविच्छेदं लिखत।",
                "विद्या + आलयः",
                "विद्यालयः = विद्या + आलयः।",
                "शब्द को दो मूल पदों में विभाजित करें।",
            ),

            (
                "देवेन्द्रः इत्यस्य सन्धिविच्छेदं लिखत।",
                "देव + इन्द्रः",
                "देवेन्द्रः = देव + इन्द्रः।",
                "दो मूल शब्द पहचानें।",
            ),

            (
                "महेशः इत्यस्य सन्धिविच्छेदं लिखत।",
                "महा + ईशः",
                "महेशः = महा + ईशः।",
                "महा और ईश शब्दों पर ध्यान दें।",
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
    # SAMASA
    # ========================================================

    def _samas_question(
        self,
        difficulty: str,
    ) -> SanskritQuestion:

        questions = [

            (
                "राजपुरुषः इत्यस्य विग्रहं लिखत।",
                "राज्ञः पुरुषः",
                "राजपुरुषः = राज्ञः पुरुषः।",
                "समस्त पद का विस्तार करें।",
            ),

            (
                "नीलकमलम् इत्यस्य विग्रहं लिखत।",
                "नीलं कमलम्",
                "नीलकमलम् = नीलं कमलम्।",
                "विशेषण और विशेष्य पहचानें।",
            ),

            (
                "यथाशक्ति इत्यस्य समासभेदः कः?",
                "अव्ययीभावः",
                "यथाशक्ति अव्ययीभाव समास का उदाहरण है।",
                "अव्यय से प्रारम्भ होने वाले समस्त पद पर ध्यान दें।",
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
    # KARAKA
    # ========================================================

    def _karaka_question(
        self,
        difficulty: str,
    ) -> SanskritQuestion:

        questions = [

            (
                "रामः पुस्तकं पठति। अत्र 'रामः' पदस्य कारकं किम्?",
                "कर्तृकारकम्",
                "जो क्रिया करता है वह कर्ता होता है।",
                "क्रिया कौन कर रहा है, यह पूछें।",
            ),

            (
                "रामः लेखन्या लिखति। अत्र 'लेखन्या' पदस्य कारकं किम्?",
                "करणकारकम्",
                "जिस साधन से क्रिया की जाती है वह करण कारक है।",
                "क्रिया किस साधन से हो रही है?",
            ),

            (
                "बालकः गुरवे नमति। अत्र 'गुरवे' पदस्य कारकं किम्?",
                "सम्प्रदानकारकम्",
                "जिसके लिए या जिसे कुछ दिया जाए, वह सम्प्रदान कारक होता है।",
                "किसे? इस प्रश्न से पहचानें।",
            ),
        ]

        return self._choice_question(
            "karaka",
            "व्याकरण",
            questions,
            difficulty,
            marks=2,
        )

    # ========================================================
    # VIBHAKTI
    # ========================================================

    def _vibhakti_question(
        self,
        difficulty: str,
    ) -> SanskritQuestion:

        questions = [

            (
                "'रामेण' इत्यत्र का विभक्तिः?",
                "तृतीया विभक्तिः",
                "रामेण तृतीया विभक्ति का रूप है।",
                "करण कारक से इसका संबंध याद करें।",
            ),

            (
                "'रामस्य' इत्यत्र का विभक्तिः?",
                "षष्ठी विभक्तिः",
                "रामस्य षष्ठी विभक्ति का रूप है।",
                "सम्बन्ध बताने वाली विभक्ति याद करें।",
            ),

            (
                "'रामे' इत्यत्र का विभक्तिः?",
                "सप्तमी विभक्तिः",
                "रामे सप्तमी विभक्ति का रूप है।",
                "अधिकरण के साथ संबंध देखें।",
            ),
        ]

        return self._choice_question(
            "vibhakti",
            "व्याकरण",
            questions,
            difficulty,
            marks=1,
        )

    # ========================================================
    # DHATU
    # ========================================================

    def _dhatu_question(
        self,
        difficulty: str,
    ) -> SanskritQuestion:

        questions = [

            (
                "'पठ्' धातोः लट् लकारे प्रथमपुरुष एकवचनरूपं लिखत।",
                "पठति",
                "पठ् धातु का लट् लकार प्रथमपुरुष एकवचन रूप 'पठति' है।",
                "वर्तमान काल का रूप याद करें।",
            ),

            (
                "'गम्' धातोः लट् लकारे प्रथमपुरुष एकवचनरूपं लिखत।",
                "गच्छति",
                "गम् धातु का वर्तमानकालीन रूप 'गच्छति' है।",
                "गम् धातु का वर्तमान रूप ध्यान से याद करें।",
            ),

            (
                "'भू' धातोः लट् लकारे प्रथमपुरुष एकवचनरूपं लिखत।",
                "भवति",
                "भू धातु का लट् लकार प्रथमपुरुष एकवचन रूप 'भवति' है।",
                "भू → भव रूप परिवर्तन याद करें।",
            ),
        ]

        return self._choice_question(
            "dhatu",
            "व्याकरण",
            questions,
            difficulty,
            marks=1,
        )

    # ========================================================
    # SHABD RUP
    # ========================================================

    def _shabdarup_question(
        self,
        difficulty: str,
    ) -> SanskritQuestion:

        questions = [

            (
                "'राम' शब्दस्य प्रथमा एकवचनरूपं लिखत।",
                "रामः",
                "राम शब्द का प्रथमा एकवचन रूप रामः है।",
                "कर्ता के एकवचन रूप को याद करें।",
            ),

            (
                "'फल' शब्दस्य प्रथमा एकवचनरूपं लिखत।",
                "फलम्",
                "फल शब्द का प्रथमा एकवचन रूप फलम् है।",
                "नपुंसकलिंग शब्दों का रूप याद करें।",
            ),

            (
                "'लता' शब्दस्य प्रथमा एकवचनरूपं लिखत।",
                "लता",
                "लता शब्द का प्रथमा एकवचन रूप लता है।",
                "स्त्रीलिंग आकारान्त शब्द पर ध्यान दें।",
            ),
        ]

        return self._choice_question(
            "shabdarup",
            "व्याकरण",
            questions,
            difficulty,
            marks=1,
        )

    # ========================================================
    # PRATYAYA
    # ========================================================

    def _pratyaya_question(
        self,
        difficulty: str,
    ) -> SanskritQuestion:

        questions = [

            (
                "पठ् + तुमुन् = ?",
                "पठितुम्",
                "पठ् धातु के साथ तुमुन् प्रत्यय से पठितुम् बनता है।",
                "तुमुन् प्रत्यय से बनने वाले तुमन्त रूप को याद करें।",
            ),

            (
                "गम् + तुमुन् = ?",
                "गन्तुम्",
                "गम् + तुमुन् से गन्तुम् बनता है।",
                "गम् धातु के परिवर्तन को ध्यान से देखें।",
            ),
        ]

        return self._choice_question(
            "pratyaya",
            "व्याकरण",
            questions,
            difficulty,
            marks=2,
        )

    # ========================================================
    # AVYAYA
    # ========================================================

    def _avyaya_question(
        self,
        difficulty: str,
    ) -> SanskritQuestion:

        questions = [

            (
                "अधोलिखितेषु अव्ययपदं चिनुत — रामः, फलम्, अत्र, बालकः",
                "अत्र",
                "अत्र एक अव्यय है।",
                "जिसका रूप नहीं बदलता उसे पहचानें।",
            ),

            (
                "'तत्र' शब्दस्य अर्थः कः?",
                "वहाँ",
                "तत्र का अर्थ 'वहाँ' होता है।",
                "स्थान बताने वाला शब्द याद करें।",
            ),

            (
                "'कुत्र' शब्दस्य अर्थः कः?",
                "कहाँ",
                "कुत्र का अर्थ 'कहाँ' होता है।",
                "प्रश्नवाचक स्थानवाचक शब्द।",
            ),
        ]

        return self._choice_question(
            "avyaya",
            "व्याकरण",
            questions,
            difficulty,
            marks=1,
        )

    # ========================================================
    # TRANSLATION
    # ========================================================

    def _translation_question(
        self,
        difficulty: str,
    ) -> SanskritQuestion:

        questions = [

            (
                "'रामः विद्यालयं गच्छति।' इत्यस्य हिन्दी-अनुवादं लिखत।",
                "राम विद्यालय जाता है।",
                "रामः = राम, विद्यालयं = विद्यालय, गच्छति = जाता है।",
                "पहले कर्ता और फिर क्रिया पहचानें।",
            ),

            (
                "'बालकाः क्रीडन्ति।' इत्यस्य हिन्दी-अनुवादं लिखत।",
                "बालक खेलते हैं।",
                "बालकाः बहुवचन है और क्रीडन्ति का अर्थ खेलते हैं।",
                "बहुवचन पर ध्यान दें।",
            ),

            (
                "'सीता पुस्तकं पठति।' इत्यस्य हिन्दी-अनुवादं लिखत।",
                "सीता पुस्तक पढ़ती है।",
                "सीता = सीता, पुस्तकं = पुस्तक, पठति = पढ़ती है।",
                "पठ् धातु का अर्थ याद करें।",
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
    # VOCABULARY
    # ========================================================

    def _vocabulary_question(
        self,
        difficulty: str,
    ) -> SanskritQuestion:

        questions = [

            (
                "'सूर्यः' शब्दस्य पर्यायवाची शब्दं लिखत।",
                "रविः",
                "सूर्यः के पर्यायवाची शब्दों में रविः, भानुः आदि आते हैं।",
                "सूर्य के संस्कृत पर्याय याद करें।",
            ),

            (
                "'दिनम्' शब्दस्य विलोमपदं लिखत।",
                "रात्रिः",
                "दिनम् का विलोम रात्रिः है।",
                "दिन और रात का संबंध याद करें।",
            ),

            (
                "'जलम्' शब्दस्य अर्थः कः?",
                "पानी",
                "जलम् का अर्थ पानी है।",
                "जल का सामान्य हिन्दी अर्थ याद करें।",
            ),

            (
                "'गृहम्' शब्दस्य अर्थः कः?",
                "घर",
                "गृहम् का अर्थ घर है।",
                "गृह = घर।",
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
    # COMPREHENSION
    # ========================================================

    def _comprehension_question(
        self,
        difficulty: str,
    ) -> SanskritQuestion:

        passage = (
            "एकः कृषकः क्षेत्रे कार्यं करोति। "
            "सः प्रतिदिनं प्रातःकाले क्षेत्रं गच्छति। "
            "सः परिश्रमेण अन्नं उत्पादयति। "
            "कृषकस्य परिश्रमः समाजस्य कृते महत्त्वपूर्णः अस्ति।"
        )

        return self._question(
            topic="comprehension",
            category="पठन",
            question=(
                f"गद्यांशं पठित्वा उत्तरं लिखत —\n\n"
                f"{passage}\n\n"
                f"कृषकः कदा क्षेत्रं गच्छति?"
            ),
            answer="प्रातःकाले",
            difficulty=difficulty,
            explanation=(
                "गद्यांश में स्पष्ट रूप से कहा गया है "
                "कि कृषकः प्रातःकाले क्षेत्रं गच्छति।"
            ),
            hint="गद्यांश की दूसरी पंक्ति देखें।",
            marks=1,
        )

    # ========================================================
    # SENTENCE
    # ========================================================

    def _sentence_question(
        self,
        difficulty: str,
    ) -> SanskritQuestion:

        questions = [

            (
                "'रामः / पठति / पुस्तकं' इत्येषां पदानां "
                "क्रमं संयोज्य शुद्धवाक्यं लिखत।",
                "रामः पुस्तकं पठति।",
                "कर्ता + कर्म + क्रिया के क्रम से वाक्य बनेगा।",
                "कर्ता पहले पहचानें।",
            ),

            (
                "'बालकाः / क्रीडन्ति / उद्याने' इत्येषां "
                "पदानां क्रमं संयोजयत।",
                "बालकाः उद्याने क्रीडन्ति।",
                "वाक्य में बालकाः कर्ता, उद्याने स्थान और क्रीडन्ति क्रिया है।",
                "कर्ता, स्थान और क्रिया को व्यवस्थित करें।",
            ),
        ]

        return self._choice_question(
            "sentence",
            "भाषा",
            questions,
            difficulty,
            marks=2,
        )

    # ========================================================
    # SUBHASHITA
    # ========================================================

    def _subhashita_question(
        self,
        difficulty: str,
    ) -> SanskritQuestion:

        questions = [

            (
                "“उद्यमेन हि सिध्यन्ति कार्याणि न मनोरथैः।” "
                "इत्यस्य भावार्थं लिखत।",
                "केवल इच्छा करने से कार्य पूर्ण नहीं होते; परिश्रम और प्रयास से ही सफलता प्राप्त होती है।",
                "इस सुभाषित का मुख्य संदेश परिश्रम का महत्व है।",
                "उद्यम का अर्थ प्रयास या परिश्रम है।",
                "उद्यम = परिश्रम",
            ),

            (
                "“विद्या ददाति विनयं” इत्यस्य भावार्थं लिखत।",
                "विद्या मनुष्य को विनम्रता प्रदान करती है।",
                "सुभाषित में शिक्षा और विनम्रता के संबंध को बताया गया है।",
                "विद्या के प्रभाव पर विचार करें।",
                "विद्या = ज्ञान/शिक्षा",
            ),
        ]

        return self._choice_question(
            "subhashita",
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
    ) -> SanskritQuestion:

        questions = [

            (
                "संस्कृत साहित्य में गद्य और पद्य में क्या अंतर है?",
                "गद्य सामान्य वाक्यात्मक रचना है जबकि पद्य छन्द, लय या काव्यात्मक शैली में रचित होता है।",
                "दोनों साहित्यिक अभिव्यक्ति के प्रमुख रूप हैं।",
                "गद्य और काव्य की रचना-शैली की तुलना करें।",
            ),

            (
                "संस्कृत साहित्य के अध्ययन से क्या लाभ होता है?",
                "भाषा, संस्कृति, साहित्यिक परंपरा और भारतीय ज्ञान परंपरा को समझने में सहायता मिलती है।",
                "संस्कृत साहित्य भाषा और संस्कृति दोनों से जुड़ा है।",
                "भाषा और संस्कृति दोनों पर विचार करें।",
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
    ) -> SanskritQuestion:

        selected = self.random.choice(
            questions
        )

        question = selected[0]
        answer = selected[1]
        explanation = selected[2]
        hint = selected[3]

        return self._question(
            topic=topic,
            category=category,
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=explanation,
            hint=hint,
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
    ) -> SanskritQuestion:

        self.question_counter += 1

        return SanskritQuestion(
            question_id=(
                f"SANS-{self.question_counter:06d}"
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
    ) -> SanskritQuestion:

        data = self.get_topic(topic)

        if data is None:

            return self._question(
                topic=topic,
                category="संस्कृत",
                question=(
                    f"'{topic}' विषय का संक्षिप्त "
                    f"विवरण लिखत।"
                ),
                answer="",
                difficulty=difficulty,
                explanation=(
                    "उत्तर में परिभाषा, मुख्य विशेषताएँ "
                    "और उदाहरण लिखें।"
                ),
                hint="पहले विषय की परिभाषा लिखें।",
                marks=3,
            )

        return self._question(
            topic=topic,
            category=data.category,
            question=(
                f"'{data.name}' विषयं "
                f"व्याख्यात।"
            ),
            answer=data.description,
            difficulty=difficulty,
            explanation=data.description,
            hint="परिभाषा से उत्तर प्रारम्भ करें।",
            marks=3,
        )

    # ========================================================
    # ANSWER CHECKING
    # ========================================================

    def check_answer(
        self,
        question: SanskritQuestion,
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
                "feedback": "उत्तर खाली है।",
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
                "feedback": "उत्तम! उत्तर पूर्णतः सही है।",
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
                    "मुख्य उत्तर सही है। "
                    "थोड़ी और सटीकता से उत्तर बेहतर होगा।"
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
                    "उत्तर में कुछ सही बातें हैं, "
                    "लेकिन मुख्य बिंदु पूरे नहीं हैं।"
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
                "उत्तर सही नहीं है। "
                "दिए गए समाधान को देखकर दोबारा प्रयास करें।"
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
    # HINTS / SOLUTIONS
    # ========================================================

    def get_hint(
        self,
        question: SanskritQuestion,
    ) -> str:

        return question.hint

    def get_solution(
        self,
        question: SanskritQuestion,
    ) -> str:

        return question.explanation

    # ========================================================
    # RANDOM QUESTION
    # ========================================================

    def generate_random_question(
        self,
        *,
        category: str | None = None,
        difficulty: str = "medium",
    ) -> SanskritQuestion:

        topic = self.random_topic(
            category=category
        )

        if topic is None:

            return self._generic_question(
                "sanskrit",
                difficulty,
            )

        return self.generate_question(
            topic.topic_id,
            difficulty=difficulty,
        )

    # ========================================================
    # RANDOM TOPIC
    # ========================================================

    def random_topic(
        self,
        *,
        category: str | None = None,
    ) -> SanskritTopic | None:

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

    # ========================================================
    # REVISION SET
    # ========================================================

    def create_revision_set(
        self,
        *,
        category: str | None = None,
        count: int = 10,
        difficulty: str = "medium",
    ) -> list[SanskritQuestion]:

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
            SanskritQuestion
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
    # PRACTICE MODES
    # ========================================================

    def practice_grammar(
        self,
        count: int = 10,
        *,
        difficulty: str = "medium",
    ) -> list[SanskritQuestion]:

        grammar_topics = [
            "sandhi",
            "samas",
            "karaka",
            "vibhakti",
            "dhatu",
            "shabdarup",
            "pratyaya",
            "avyaya",
        ]

        questions = []

        for _ in range(max(0, count)):

            topic = self.random.choice(
                grammar_topics
            )

            questions.append(
                self.generate_question(
                    topic,
                    difficulty=difficulty,
                )
            )

        return questions

    def practice_translation(
        self,
        count: int = 10,
        *,
        difficulty: str = "medium",
    ) -> list[SanskritQuestion]:

        return [
            self.generate_question(
                "translation",
                difficulty=difficulty,
            )
            for _ in range(max(0, count))
        ]

    def practice_vocabulary(
        self,
        count: int = 10,
        *,
        difficulty: str = "easy",
    ) -> list[SanskritQuestion]:

        return [
            self.generate_question(
                "vocabulary",
                difficulty=difficulty,
            )
            for _ in range(max(0, count))
        ]

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
            "व्याख्या",
            "विश्लेषण",
            "तुलना",
            "भेद",
            "विस्तार",
            "भावार्थ",
            "विग्रह",
            "सन्धिविच्छेद",
            "समास",
            "रूप",
        ]

        easy_terms = [
            "क्या",
            "कौन",
            "कहाँ",
            "कब",
            "अर्थ",
            "नाम",
            "चिनुत",
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
    # RECOMMENDATIONS
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

    # ========================================================
    # DICTIONARY
    # ========================================================

    def get_basic_dictionary(self) -> dict[str, str]:

        return {
            "सूर्यः": "सूरज",
            "चन्द्रः": "चन्द्रमा",
            "जलम्": "पानी",
            "अग्निः": "आग",
            "गृहम्": "घर",
            "विद्यालयः": "विद्यालय",
            "पुस्तकम्": "किताब",
            "मित्रम्": "मित्र",
            "गुरुः": "शिक्षक",
            "बालकः": "लड़का",
            "बालिका": "लड़की",
            "वृक्षः": "पेड़",
            "पुष्पम्": "फूल",
            "फलम्": "फल",
            "मार्गः": "रास्ता",
            "नदी": "नदी",
            "पर्वतः": "पहाड़",
            "दिनम्": "दिन",
            "रात्रिः": "रात",
            "ज्ञानम्": "ज्ञान",
            "विद्या": "शिक्षा",
            "शान्तिः": "शांति",
            "प्रेम": "प्रेम",
            "सत्य": "सत्य",
            "धर्मः": "धर्म",
            "मित्रम्": "दोस्त",
        }

    def word_meaning(
        self,
        word: str,
    ) -> str | None:

        dictionary = self.get_basic_dictionary()

        normalized = word.strip()

        return dictionary.get(
            normalized
        )

    # ========================================================
    # SHLOKA / BHAVARTH
    # ========================================================

    def explain_shloka(
        self,
        shloka: str,
    ) -> dict[str, str]:

        text = self._normalize(
            shloka
        )

        if "उद्यमेन" in text:

            return {
                "shloka": shloka,
                "meaning": (
                    "परिश्रम करने से ही कार्य सफल होते हैं; "
                    "केवल इच्छा करने से नहीं।"
                ),
                "message": (
                    "जीवन में सफलता के लिए निरंतर प्रयास आवश्यक है।"
                ),
            }

        if "विद्या ददाति विनयम्" in text:

            return {
                "shloka": shloka,
                "meaning": (
                    "विद्या मनुष्य को विनम्रता प्रदान करती है।"
                ),
                "message": (
                    "सच्चा ज्ञान व्यक्ति में विनम्रता और अच्छे गुण विकसित करता है।"
                ),
            }

        return {
            "shloka": shloka,
            "meaning": (
                "श्लोक का अर्थ संदर्भ और शब्दार्थ के आधार पर समझाया जाना चाहिए।"
            ),
            "message": (
                "पहले कठिन शब्दों के अर्थ समझें और फिर पूरे श्लोक का भावार्थ लिखें।"
            ),
        }

    # ========================================================
    # UTILITY
    # ========================================================

    @staticmethod
    def _normalize(
        value: str,
    ) -> str:

        value = str(value).casefold()

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


# ============================================================
# FACTORY
# ============================================================


def create_sanskrit_engine(
    *,
    seed: int | None = None,
) -> SanskritEngine:

    return SanskritEngine(
        seed=seed
    )


# ============================================================
# EXPORTS
# ============================================================


__all__ = [
    "SanskritTopic",
    "SanskritQuestion",
    "SanskritEngine",
    "create_sanskrit_engine",
]


