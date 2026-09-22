"""
RENIX Education — Hindi Subject Engine

File:
    RENIX/education/subjects/hindi.py

Features:
- Hindi grammar
- Vocabulary
- पर्यायवाची शब्द
- विलोम शब्द
- मुहावरे
- संधि
- समास
- अलंकार
- काल
- वचन
- लिंग
- कारक
- वाक्य शुद्धि
- Reading comprehension
- Writing practice
- Literature questions
- Question generation
- Answer checking
- Hints and explanations
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
class HindiTopic:
    topic_id: str
    name: str
    category: str
    description: str
    difficulty: str = "medium"
    skills: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "difficulty": self.difficulty,
            "skills": list(self.skills),
        }


@dataclass
class HindiQuestion:
    question_id: str
    topic: str
    category: str
    question: str
    answer: Any
    difficulty: str = "medium"
    explanation: str = ""
    hint: str = ""
    options: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "topic": self.topic,
            "category": self.category,
            "question": self.question,
            "answer": self.answer,
            "difficulty": self.difficulty,
            "explanation": self.explanation,
            "hint": self.hint,
            "options": list(self.options),
        }


# ============================================================
# HINDI ENGINE
# ============================================================


class HindiEngine:

    def __init__(
        self,
        *,
        seed: int | None = None,
    ) -> None:

        self.random = random.Random(seed)

        self.topics: dict[str, HindiTopic] = {}

        self.question_counter = 0

        self._register_default_topics()

    # ========================================================
    # TOPICS
    # ========================================================

    def _register_default_topics(self) -> None:

        topics = [

            HindiTopic(
                topic_id="sangya",
                name="संज्ञा",
                category="व्याकरण",
                description=(
                    "व्यक्ति, वस्तु, स्थान, भाव आदि के नाम "
                    "बताने वाले शब्दों का अध्ययन।"
                ),
                difficulty="easy",
                skills=["व्याकरण", "पहचान"],
            ),

            HindiTopic(
                topic_id="sarvanam",
                name="सर्वनाम",
                category="व्याकरण",
                description=(
                    "संज्ञा के स्थान पर प्रयोग किए जाने वाले शब्द।"
                ),
                difficulty="easy",
                skills=["व्याकरण", "वाक्य निर्माण"],
            ),

            HindiTopic(
                topic_id="visheshan",
                name="विशेषण",
                category="व्याकरण",
                description=(
                    "संज्ञा या सर्वनाम की विशेषता बताने वाले शब्द।"
                ),
                difficulty="easy",
                skills=["व्याकरण", "पहचान"],
            ),

            HindiTopic(
                topic_id="kriya",
                name="क्रिया",
                category="व्याकरण",
                description=(
                    "कार्य या अवस्था का बोध कराने वाले शब्द।"
                ),
                difficulty="easy",
                skills=["व्याकरण", "वाक्य विश्लेषण"],
            ),

            HindiTopic(
                topic_id="kaal",
                name="काल",
                category="व्याकरण",
                description=(
                    "क्रिया के समय को बताने वाले रूप।"
                ),
                difficulty="medium",
                skills=["व्याकरण", "वाक्य परिवर्तन"],
            ),

            HindiTopic(
                topic_id="ling",
                name="लिंग",
                category="व्याकरण",
                description=(
                    "पुल्लिंग और स्त्रीलिंग शब्दों का अध्ययन।"
                ),
                difficulty="easy",
                skills=["व्याकरण", "शब्द ज्ञान"],
            ),

            HindiTopic(
                topic_id="vachan",
                name="वचन",
                category="व्याकरण",
                description=(
                    "एकवचन और बहुवचन का अध्ययन।"
                ),
                difficulty="easy",
                skills=["व्याकरण", "वाक्य निर्माण"],
            ),

            HindiTopic(
                topic_id="karak",
                name="कारक",
                category="व्याकरण",
                description=(
                    "संज्ञा या सर्वनाम का वाक्य के अन्य शब्दों "
                    "से संबंध बताने वाले कारक।"
                ),
                difficulty="medium",
                skills=["व्याकरण", "वाक्य विश्लेषण"],
            ),

            HindiTopic(
                topic_id="sandhi",
                name="संधि",
                category="शब्द-विचार",
                description=(
                    "दो वर्णों या शब्दों के मेल से होने वाले "
                    "परिवर्तन का अध्ययन।"
                ),
                difficulty="hard",
                skills=["शब्द-विचार", "विश्लेषण"],
            ),

            HindiTopic(
                topic_id="samas",
                name="समास",
                category="शब्द-विचार",
                description=(
                    "दो या अधिक शब्दों के संक्षिप्त रूप में "
                    "निर्माण का अध्ययन।"
                ),
                difficulty="hard",
                skills=["शब्द-विचार", "विग्रह"],
            ),

            HindiTopic(
                topic_id="paryayvachi",
                name="पर्यायवाची शब्द",
                category="शब्द ज्ञान",
                description=(
                    "समान या मिलते-जुलते अर्थ वाले शब्द।"
                ),
                difficulty="easy",
                skills=["शब्द ज्ञान", "स्मरण"],
            ),

            HindiTopic(
                topic_id="vilom",
                name="विलोम शब्द",
                category="शब्द ज्ञान",
                description=(
                    "एक-दूसरे के विपरीत अर्थ वाले शब्द।"
                ),
                difficulty="easy",
                skills=["शब्द ज्ञान", "स्मरण"],
            ),

            HindiTopic(
                topic_id="muhavare",
                name="मुहावरे",
                category="शब्द ज्ञान",
                description=(
                    "विशिष्ट अर्थ देने वाले प्रचलित वाक्यांश।"
                ),
                difficulty="medium",
                skills=["भाषा", "अर्थ-बोध"],
            ),

            HindiTopic(
                topic_id="alankar",
                name="अलंकार",
                category="काव्य",
                description=(
                    "काव्य की सुंदरता बढ़ाने वाले भाषिक उपकरण।"
                ),
                difficulty="hard",
                skills=["काव्य", "पहचान", "विश्लेषण"],
            ),

            HindiTopic(
                topic_id="apathit_gadyansh",
                name="अपठित गद्यांश",
                category="पठन",
                description=(
                    "गद्यांश पढ़कर प्रश्नों के उत्तर देना।"
                ),
                difficulty="medium",
                skills=["पठन", "बोध", "विश्लेषण"],
            ),

            HindiTopic(
                topic_id="lekhan",
                name="लेखन कौशल",
                category="लेखन",
                description=(
                    "पत्र, अनुच्छेद, निबंध, भाषण और अन्य "
                    "रचनात्मक लेखन का अभ्यास।"
                ),
                difficulty="hard",
                skills=["लेखन", "भाषा", "रचनात्मकता"],
            ),

            HindiTopic(
                topic_id="sahitya",
                name="हिंदी साहित्य",
                category="साहित्य",
                description=(
                    "गद्य, पद्य, लेखक, कवि, पात्र, भावार्थ "
                    "और साहित्यिक विश्लेषण।"
                ),
                difficulty="hard",
                skills=["साहित्य", "विश्लेषण"],
            ),
        ]

        for topic in topics:
            self.register_topic(topic)

    def register_topic(
        self,
        topic: HindiTopic,
    ) -> None:

        self.topics[
            self._normalize_topic(topic.name)
        ] = topic

    # ========================================================
    # TOPIC ACCESS
    # ========================================================

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

    def get_topic(
        self,
        topic: str,
    ) -> HindiTopic | None:

        return self.topics.get(
            self._normalize_topic(topic)
        )

    def search_topics(
        self,
        query: str,
    ) -> list[dict[str, Any]]:

        query = query.casefold().strip()

        results = []

        for topic in self.topics.values():

            searchable = (
                f"{topic.name} "
                f"{topic.category} "
                f"{topic.description}"
            ).casefold()

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
    ) -> HindiQuestion:

        normalized = self._normalize_topic(topic)

        generators = {
            "sangya": self._sangya_question,
            "sarvanam": self._sarvanam_question,
            "visheshan": self._visheshan_question,
            "kriya": self._kriya_question,
            "kaal": self._kaal_question,
            "ling": self._ling_question,
            "vachan": self._vachan_question,
            "karak": self._karak_question,
            "sandhi": self._sandhi_question,
            "samas": self._samas_question,
            "paryayvachi": self._paryayvachi_question,
            "vilom": self._vilom_question,
            "muhavare": self._muhavara_question,
            "alankar": self._alankar_question,
            "apathit_gadyansh": self._reading_question,
            "lekhan": self._writing_question,
            "sahitya": self._literature_question,
        }

        generator = generators.get(normalized)

        if generator is None:
            return self._generic_question(
                normalized,
                difficulty,
            )

        return generator(difficulty)

    def generate_questions(
        self,
        topic: str,
        count: int = 10,
        *,
        difficulty: str = "medium",
    ) -> list[HindiQuestion]:

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
    # SANGYA
    # ========================================================

    def _sangya_question(
        self,
        difficulty: str,
    ) -> HindiQuestion:

        questions = [
            (
                "वाक्य में संज्ञा पहचानिए: 'राम बाजार गया।'",
                "राम, बाजार",
                "राम व्यक्ति का नाम और बाजार स्थान का नाम है।",
                ["राम, बाजार", "गया", "राम गया", "कोई नहीं"],
            ),
            (
                "इनमें से संज्ञा कौन-सा शब्द है?",
                "पुस्तक",
                "'पुस्तक' एक वस्तु का नाम है।",
                ["सुंदर", "पुस्तक", "चलना", "वह"],
            ),
        ]

        question, answer, explanation, options = (
            self.random.choice(questions)
        )

        return self._question(
            topic="sangya",
            category="व्याकरण",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=explanation,
            hint="व्यक्ति, वस्तु, स्थान या भाव के नाम को पहचानें।",
            options=options,
        )

    # ========================================================
    # SARVANAM
    # ========================================================

    def _sarvanam_question(
        self,
        difficulty: str,
    ) -> HindiQuestion:

        questions = [
            (
                "वाक्य में सर्वनाम पहचानिए: 'वह स्कूल जा रहा है।'",
                "वह",
                "'वह' संज्ञा के स्थान पर प्रयोग किया गया है।",
                ["वह", "स्कूल", "जा", "रहा"],
            ),
            (
                "इनमें से सर्वनाम कौन-सा है?",
                "मैं",
                "'मैं' बोलने वाले व्यक्ति के लिए प्रयोग होने वाला सर्वनाम है।",
                ["घर", "सुंदर", "मैं", "दौड़ना"],
            ),
        ]

        question, answer, explanation, options = (
            self.random.choice(questions)
        )

        return self._question(
            topic="sarvanam",
            category="व्याकरण",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=explanation,
            hint="देखें कि कौन-सा शब्द संज्ञा के स्थान पर आया है।",
            options=options,
        )

    # ========================================================
    # VISHESHAN
    # ========================================================

    def _visheshan_question(
        self,
        difficulty: str,
    ) -> HindiQuestion:

        questions = [
            (
                "वाक्य में विशेषण पहचानिए: 'मोहन ने लाल गेंद खरीदी।'",
                "लाल",
                "'लाल' गेंद की विशेषता बता रहा है।",
                ["मोहन", "ने", "लाल", "खरीदी"],
            ),
            (
                "वाक्य में विशेषण पहचानिए: 'सुंदर फूल खिल रहे हैं।'",
                "सुंदर",
                "'सुंदर' फूल की विशेषता बताता है।",
                ["फूल", "सुंदर", "खिल", "रहे"],
            ),
        ]

        question, answer, explanation, options = (
            self.random.choice(questions)
        )

        return self._question(
            topic="visheshan",
            category="व्याकरण",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=explanation,
            hint="जो शब्द किसी संज्ञा की विशेषता बताए उसे खोजें।",
            options=options,
        )

    # ========================================================
    # KRIYA
    # ========================================================

    def _kriya_question(
        self,
        difficulty: str,
    ) -> HindiQuestion:

        questions = [
            (
                "वाक्य में क्रिया पहचानिए: 'राहुल क्रिकेट खेलता है।'",
                "खेलता है",
                "यह शब्द कार्य का बोध कराता है।",
                ["राहुल", "क्रिकेट", "खेलता है", "है"],
            ),
            (
                "वाक्य में क्रिया पहचानिए: 'सीमा पुस्तक पढ़ रही है।'",
                "पढ़ रही है",
                "यह वाक्य में किए जा रहे कार्य को दर्शाता है।",
                ["सीमा", "पुस्तक", "पढ़ रही है", "कोई नहीं"],
            ),
        ]

        question, answer, explanation, options = (
            self.random.choice(questions)
        )

        return self._question(
            topic="kriya",
            category="व्याकरण",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=explanation,
            hint="कार्य या अवस्था बताने वाले शब्द को खोजें।",
            options=options,
        )

    # ========================================================
    # KAAL
    # ========================================================

    def _kaal_question(
        self,
        difficulty: str,
    ) -> HindiQuestion:

        questions = [
            (
                "वाक्य का काल बताइए: 'मैं कल स्कूल गया था।'",
                "भूतकाल",
                "'गया था' बीते हुए समय का बोध कराता है।",
                ["वर्तमानकाल", "भूतकाल", "भविष्यत्काल"],
            ),
            (
                "वाक्य का काल बताइए: 'मैं रोज पढ़ता हूँ।'",
                "वर्तमानकाल",
                "'पढ़ता हूँ' वर्तमान समय की आदत दर्शाता है।",
                ["वर्तमानकाल", "भूतकाल", "भविष्यत्काल"],
            ),
            (
                "वाक्य का काल बताइए: 'मैं कल मुंबई जाऊँगा।'",
                "भविष्यत्काल",
                "'जाऊँगा' भविष्य में होने वाले कार्य को दर्शाता है।",
                ["वर्तमानकाल", "भूतकाल", "भविष्यत्काल"],
            ),
        ]

        question, answer, explanation, options = (
            self.random.choice(questions)
        )

        return self._question(
            topic="kaal",
            category="व्याकरण",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=explanation,
            hint="क्रिया किस समय हो रही है, यह देखें।",
            options=options,
        )

    # ========================================================
    # LING
    # ========================================================

    def _ling_question(
        self,
        difficulty: str,
    ) -> HindiQuestion:

        questions = [
            (
                "'लड़का' का स्त्रीलिंग क्या है?",
                "लड़की",
                "'लड़का' का स्त्रीलिंग 'लड़की' है।",
                ["लड़के", "लड़की", "लड़कियाँ", "बालक"],
            ),
            (
                "'राजा' का स्त्रीलिंग क्या है?",
                "रानी",
                "'राजा' का स्त्रीलिंग 'रानी' है।",
                ["राजी", "रानी", "राजकुमार", "राजा"],
            ),
        ]

        question, answer, explanation, options = (
            self.random.choice(questions)
        )

        return self._question(
            topic="ling",
            category="व्याकरण",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=explanation,
            hint="शब्द के स्त्री रूप पर ध्यान दें।",
            options=options,
        )

    # ========================================================
    # VACHAN
    # ========================================================

    def _vachan_question(
        self,
        difficulty: str,
    ) -> HindiQuestion:

        questions = [
            (
                "'लड़का' का बहुवचन क्या है?",
                "लड़के",
                "'लड़का' का सामान्य बहुवचन 'लड़के' है।",
                ["लड़की", "लड़के", "लड़कों", "लड़का"],
            ),
            (
                "'पुस्तक' का बहुवचन क्या है?",
                "पुस्तकें",
                "'पुस्तक' का बहुवचन 'पुस्तकें' है।",
                ["पुस्तकी", "पुस्तकें", "पुस्तकों", "पुस्तक"],
            ),
        ]

        question, answer, explanation, options = (
            self.random.choice(questions)
        )

        return self._question(
            topic="vachan",
            category="व्याकरण",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=explanation,
            hint="एक से अधिक वस्तुओं या व्यक्तियों का रूप सोचें।",
            options=options,
        )

    # ========================================================
    # KARAK
    # ========================================================

    def _karak_question(
        self,
        difficulty: str,
    ) -> HindiQuestion:

        questions = [
            (
                "'राम ने खाना खाया।' में 'ने' कौन-सा कारक है?",
                "कर्ता कारक",
                "'ने' कर्ता का संबंध बताता है।",
                [
                    "कर्ता कारक",
                    "कर्म कारक",
                    "करण कारक",
                    "अधिकरण कारक",
                ],
            ),
            (
                "'मोहन ने चाकू से फल काटा।' में 'से' कौन-सा कारक है?",
                "करण कारक",
                "'से' साधन का बोध कराता है।",
                [
                    "कर्ता कारक",
                    "करण कारक",
                    "संबंध कारक",
                    "संबोधन कारक",
                ],
            ),
        ]

        question, answer, explanation, options = (
            self.random.choice(questions)
        )

        return self._question(
            topic="karak",
            category="व्याकरण",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=explanation,
            hint="कारक-चिह्न और उसके संबंध को पहचानें।",
            options=options,
        )

    # ========================================================
    # SANDHI
    # ========================================================

    def _sandhi_question(
        self,
        difficulty: str,
    ) -> HindiQuestion:

        questions = [
            (
                "'विद्यालय' का संधि-विच्छेद कीजिए।",
                "विद्या + आलय",
                "'विद्यालय' = विद्या + आलय।",
            ),
            (
                "'महेश' का संधि-विच्छेद कीजिए।",
                "महा + ईश",
                "'महेश' का संधि-विच्छेद 'महा + ईश' माना जाता है।",
            ),
        ]

        question, answer, explanation = self.random.choice(
            questions
        )

        return self._question(
            topic="sandhi",
            category="शब्द-विचार",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=explanation,
            hint="शब्द को उसके मूल शब्दों में विभाजित करें।",
        )

    # ========================================================
    # SAMAS
    # ========================================================

    def _samas_question(
        self,
        difficulty: str,
    ) -> HindiQuestion:

        questions = [
            (
                "'राजपुत्र' का समास-विग्रह कीजिए।",
                "राजा का पुत्र",
                "राजपुत्र = राजा का पुत्र।",
            ),
            (
                "'नीलकमल' का समास-विग्रह कीजिए।",
                "नीला है जो कमल",
                "नीलकमल में 'नीला' कमल की विशेषता बताता है।",
            ),
        ]

        question, answer, explanation = self.random.choice(
            questions
        )

        return self._question(
            topic="samas",
            category="शब्द-विचार",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=explanation,
            hint="समस्त पद के दोनों शब्दों के संबंध को स्पष्ट करें।",
        )

    # ========================================================
    # PARYAYVACHI
    # ========================================================

    def _paryayvachi_question(
        self,
        difficulty: str,
    ) -> HindiQuestion:

        questions = [
            (
                "'सूर्य' का पर्यायवाची शब्द चुनिए।",
                "रवि",
                "'रवि' सूर्य का पर्यायवाची है।",
                ["रवि", "रात्रि", "पवन", "जल"],
            ),
            (
                "'जल' का पर्यायवाची शब्द चुनिए।",
                "पानी",
                "'पानी' जल का सामान्य पर्यायवाची है।",
                ["अग्नि", "पानी", "आकाश", "धरती"],
            ),
            (
                "'पृथ्वी' का पर्यायवाची शब्द चुनिए।",
                "धरती",
                "'धरती' पृथ्वी का पर्यायवाची है।",
                ["सूर्य", "धरती", "बादल", "समुद्र"],
            ),
        ]

        question, answer, explanation, options = (
            self.random.choice(questions)
        )

        return self._question(
            topic="paryayvachi",
            category="शब्द ज्ञान",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=explanation,
            hint="समान अर्थ वाला शब्द खोजें।",
            options=options,
        )

    # ========================================================
    # VILOM
    # ========================================================

    def _vilom_question(
        self,
        difficulty: str,
    ) -> HindiQuestion:

        questions = [
            (
                "'दिन' का विलोम शब्द क्या है?",
                "रात",
                "'दिन' और 'रात' विपरीत अर्थ वाले शब्द हैं।",
                ["सुबह", "रात", "दोपहर", "सूरज"],
            ),
            (
                "'सत्य' का विलोम शब्द क्या है?",
                "असत्य",
                "'असत्य' सत्य का विपरीत अर्थ देता है।",
                ["सही", "असत्य", "न्याय", "धर्म"],
            ),
            (
                "'लाभ' का विलोम शब्द क्या है?",
                "हानि",
                "'हानि' लाभ का विलोम है।",
                ["धन", "हानि", "जीत", "सफलता"],
            ),
        ]

        question, answer, explanation, options = (
            self.random.choice(questions)
        )

        return self._question(
            topic="vilom",
            category="शब्द ज्ञान",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=explanation,
            hint="विपरीत अर्थ वाला शब्द चुनें।",
            options=options,
        )

    # ========================================================
    # MUHAVARE
    # ========================================================

    def _muhavara_question(
        self,
        difficulty: str,
    ) -> HindiQuestion:

        questions = [
            (
                "'आँखों का तारा' मुहावरे का अर्थ क्या है?",
                "बहुत प्यारा व्यक्ति",
                "आँखों का तारा किसी बहुत प्रिय व्यक्ति के लिए कहा जाता है।",
            ),
            (
                "'नाक में दम करना' का अर्थ क्या है?",
                "बहुत परेशान करना",
                "किसी को बहुत अधिक परेशान करना।",
            ),
            (
                "'हाथ पर हाथ धरे बैठना' का अर्थ क्या है?",
                "कुछ न करना",
                "निष्क्रिय होकर बैठे रहना।",
            ),
        ]

        question, answer, explanation = self.random.choice(
            questions
        )

        return self._question(
            topic="muhavare",
            category="शब्द ज्ञान",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=explanation,
            hint="मुहावरे का भावार्थ सोचें, शाब्दिक अर्थ नहीं।",
        )

    # ========================================================
    # ALANKAR
    # ========================================================

    def _alankar_question(
        self,
        difficulty: str,
    ) -> HindiQuestion:

        questions = [
            (
                "'चंचल चितवन चुरा रही' में कौन-सा अलंकार है?",
                "अनुप्रास अलंकार",
                "'च' ध्वनि की पुनरावृत्ति के कारण अनुप्रास है।",
                [
                    "उपमा",
                    "रूपक",
                    "अनुप्रास अलंकार",
                    "मानवीकरण",
                ],
            ),
            (
                "'मुख चंद्रमा के समान सुंदर है।' में कौन-सा अलंकार है?",
                "उपमा अलंकार",
                "'समान' शब्द द्वारा तुलना की गई है।",
                [
                    "उपमा अलंकार",
                    "रूपक",
                    "अनुप्रास",
                    "यमक",
                ],
            ),
            (
                "'जीवन एक यात्रा है।' में कौन-सा अलंकार है?",
                "रूपक अलंकार",
                "जीवन को सीधे यात्रा कहा गया है।",
                [
                    "रूपक अलंकार",
                    "उपमा",
                    "अनुप्रास",
                    "श्लेष",
                ],
            ),
        ]

        question, answer, explanation, options = (
            self.random.choice(questions)
        )

        return self._question(
            topic="alankar",
            category="काव्य",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=explanation,
            hint="तुलना, ध्वनि की पुनरावृत्ति या सीधा रूपांतरण देखें।",
            options=options,
        )

    # ========================================================
    # READING
    # ========================================================

    def _reading_question(
        self,
        difficulty: str,
    ) -> HindiQuestion:

        passages = [

            (
                (
                    "परिश्रम मनुष्य को सफलता की ओर ले जाता है। "
                    "केवल प्रतिभा होना पर्याप्त नहीं है। "
                    "नियमित अभ्यास, धैर्य और आत्मविश्वास के "
                    "माध्यम से व्यक्ति अपनी कमियों को सुधार "
                    "सकता है और अपने लक्ष्य तक पहुँच सकता है।"
                ),
                "गद्यांश के अनुसार सफलता के लिए किन गुणों की आवश्यकता है?",
                "परिश्रम, नियमित अभ्यास, धैर्य और आत्मविश्वास।",
            ),

            (
                (
                    "पेड़ हमारे जीवन के लिए अत्यंत महत्वपूर्ण हैं। "
                    "वे हमें ऑक्सीजन देते हैं, वातावरण को ठंडा "
                    "रखते हैं और अनेक जीवों को आश्रय प्रदान करते हैं। "
                    "इसलिए पेड़ों की रक्षा करना हमारा कर्तव्य है।"
                ),
                "पेड़ हमारे लिए क्यों महत्वपूर्ण हैं?",
                "वे ऑक्सीजन देते हैं, वातावरण को ठंडा रखते हैं और जीवों को आश्रय देते हैं।",
            ),
        ]

        passage, question, answer = self.random.choice(
            passages
        )

        return self._question(
            topic="apathit_gadyansh",
            category="पठन",
            question=(
                f"गद्यांश पढ़िए:\n\n"
                f"{passage}\n\n"
                f"{question}"
            ),
            answer=answer,
            difficulty=difficulty,
            explanation=(
                "उत्तर गद्यांश में दी गई जानकारी के आधार पर होना चाहिए।"
            ),
            hint="गद्यांश में सीधे दिए गए मुख्य बिंदु खोजें।",
        )

    # ========================================================
    # WRITING
    # ========================================================

    def _writing_question(
        self,
        difficulty: str,
    ) -> HindiQuestion:

        prompts = [
            "विद्यालय में खेल दिवस के आयोजन पर एक रिपोर्ट लिखिए।",
            "समय का महत्व विषय पर एक अनुच्छेद लिखिए।",
            "पर्यावरण संरक्षण विषय पर भाषण लिखिए।",
            "अपने मित्र को परीक्षा की तैयारी के बारे में पत्र लिखिए।",
            "मोबाइल फोन के लाभ और हानियाँ विषय पर निबंध लिखिए।",
        ]

        prompt = self.random.choice(prompts)

        return self._question(
            topic="lekhan",
            category="लेखन",
            question=prompt,
            answer="लेखन कार्य",
            difficulty=difficulty,
            explanation=(
                "उत्तर में उचित प्रारूप, स्पष्ट विचार, "
                "सही व्याकरण, क्रमबद्ध प्रस्तुति और प्रभावी भाषा होनी चाहिए।"
            ),
            hint=(
                "पहले मुख्य बिंदु लिखें, फिर उन्हें क्रमबद्ध "
                "रूप से विस्तार दें।"
            ),
        )

    # ========================================================
    # LITERATURE
    # ========================================================

    def _literature_question(
        self,
        difficulty: str,
    ) -> HindiQuestion:

        questions = [
            (
                "किसी कविता का 'भावार्थ' क्या होता है?",
                "कविता में व्यक्त मुख्य भाव और विचार को अपने शब्दों में स्पष्ट करना।",
            ),
            (
                "कहानी में 'पात्र' किसे कहते हैं?",
                "कहानी में भाग लेने वाले व्यक्ति या चरित्र को पात्र कहते हैं।",
            ),
            (
                "किसी रचना का 'मुख्य विचार' क्या होता है?",
                "रचना के केंद्र में मौजूद प्रमुख विचार या संदेश।",
            ),
        ]

        question, answer = self.random.choice(
            questions
        )

        return self._question(
            topic="sahitya",
            category="साहित्य",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=answer,
            hint="रचना के मुख्य भाव या विचार पर ध्यान दें।",
        )

    # ========================================================
    # GENERIC QUESTION
    # ========================================================

    def _generic_question(
        self,
        topic: str,
        difficulty: str,
    ) -> HindiQuestion:

        topic_data = self.get_topic(topic)

        if topic_data is None:

            return self._question(
                topic=topic,
                category="हिंदी",
                question=(
                    f"'{topic.replace('_', ' ')}' "
                    "की परिभाषा और उदाहरण लिखिए।"
                ),
                answer="",
                difficulty=difficulty,
                explanation=(
                    "परिभाषा, मुख्य विशेषताएँ और उदाहरण लिखें।"
                ),
                hint="पहले परिभाषा लिखें और फिर एक उदाहरण दें।",
            )

        return self._question(
            topic=topic,
            category=topic_data.category,
            question=(
                f"'{topic_data.name}' की परिभाषा "
                "अपने शब्दों में लिखिए।"
            ),
            answer="conceptual",
            difficulty=difficulty,
            explanation=topic_data.description,
            hint="परिभाषा के साथ एक उदाहरण भी दें।",
        )

    # ========================================================
    # ANSWER CHECKING
    # ========================================================

    def check_answer(
        self,
        question: HindiQuestion,
        user_answer: Any,
    ) -> dict[str, Any]:

        expected = self._normalize_answer(
            question.answer
        )

        actual = self._normalize_answer(
            user_answer
        )

        if expected in {
            "लेखन कार्य",
            "conceptual",
        }:
            if expected == "लेखन कार्य":
                return self._evaluate_writing(
                    question,
                    str(user_answer),
                )

            return self._evaluate_conceptual(
                question,
                str(user_answer),
            )

        correct = (
            actual == expected
            or self._semantic_match(
                actual,
                expected,
            )
        )

        return {
            "correct": correct,
            "question_id": question.question_id,
            "topic": question.topic,
            "category": question.category,
            "user_answer": user_answer,
            "correct_answer": question.answer,
            "accuracy": 100.0 if correct else 0.0,
            "explanation": question.explanation,
            "hint": (
                None
                if correct
                else question.hint
            ),
        }

    # ========================================================
    # WRITING EVALUATION
    # ========================================================

    def _evaluate_writing(
        self,
        question: HindiQuestion,
        answer: str,
    ) -> dict[str, Any]:

        words = self._word_count(answer)

        sentences = self._sentence_count(answer)

        score = 0.0

        if words >= 100:
            score += 30
        elif words >= 60:
            score += 25
        elif words >= 30:
            score += 15
        elif words > 0:
            score += 8

        if sentences >= 6:
            score += 25
        elif sentences >= 3:
            score += 20
        elif sentences >= 1:
            score += 10

        if answer.strip():
            score += 20

        if "।" in answer:
            score += 10

        unique_words = len(
            set(
                re.findall(
                    r"[\u0900-\u097F]+",
                    answer,
                )
            )
        )

        score += min(
            15,
            unique_words * 0.5,
        )

        score = min(
            100.0,
            score,
        )

        feedback = []

        if words < 60:
            feedback.append(
                "उत्तर में अधिक विस्तार और उदाहरण जोड़ें।"
            )

        if sentences < 3:
            feedback.append(
                "विचारों को पूर्ण वाक्यों में लिखें।"
            )

        if score >= 80:
            feedback.append(
                "बहुत अच्छा लेखन।"
            )
        elif score >= 60:
            feedback.append(
                "अच्छा प्रयास। भाषा और प्रस्तुति में और सुधार करें।"
            )
        else:
            feedback.append(
                "उत्तर को अधिक व्यवस्थित और विस्तृत बनाइए।"
            )

        return {
            "correct": score >= 60,
            "question_id": question.question_id,
            "topic": question.topic,
            "category": question.category,
            "user_answer": answer,
            "correct_answer": "खुला लेखन कार्य",
            "accuracy": round(score, 2),
            "word_count": words,
            "sentence_count": sentences,
            "feedback": feedback,
            "explanation": question.explanation,
            "hint": question.hint,
        }

    # ========================================================
    # CONCEPTUAL EVALUATION
    # ========================================================

    def _evaluate_conceptual(
        self,
        question: HindiQuestion,
        answer: str,
    ) -> dict[str, Any]:

        if not answer.strip():
            score = 0.0
        else:
            words = re.findall(
                r"[\u0900-\u097F]+",
                answer,
            )

            score = min(
                100.0,
                35.0 + len(set(words)) * 4.0,
            )

        return {
            "correct": score >= 60,
            "question_id": question.question_id,
            "topic": question.topic,
            "category": question.category,
            "user_answer": answer,
            "accuracy": round(score, 2),
            "feedback": (
                "परिभाषा के साथ उदाहरण और मुख्य बिंदु जोड़ें।"
            ),
            "explanation": question.explanation,
            "hint": question.hint,
        }

    # ========================================================
    # HINT / SOLUTION
    # ========================================================

    def get_hint(
        self,
        question: HindiQuestion,
    ) -> str:

        return question.hint

    def get_solution(
        self,
        question: HindiQuestion,
    ) -> str:

        return question.explanation

    # ========================================================
    # DIFFICULTY
    # ========================================================

    def classify_difficulty(
        self,
        question: str,
    ) -> str:

        text = question.casefold()

        hard_patterns = [
            "विश्लेषण",
            "विश्लेषित",
            "तुलना",
            "व्याख्या",
            "औचित्य",
            "भावार्थ",
            "समास-विग्रह",
            "संधि-विच्छेद",
            "निबंध",
            "भाषण",
            "रिपोर्ट",
            "पत्र",
        ]

        easy_patterns = [
            "क्या है",
            "कौन",
            "पहचानिए",
            "चुनिए",
            "पर्यायवाची",
            "विलोम",
            "अर्थ",
            "परिभाषा",
        ]

        if any(
            pattern in text
            for pattern in hard_patterns
        ):
            return "hard"

        if any(
            pattern in text
            for pattern in easy_patterns
        ):
            return "easy"

        return "medium"

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    def recommend_topics(
        self,
        *,
        category: str | None = None,
        weak_topics: Iterable[str] | None = None,
        count: int = 3,
    ) -> list[str]:

        if weak_topics:
            return list(weak_topics)[:count]

        topics = list(self.topics.values())

        if category:

            category = category.casefold()

            topics = [
                topic
                for topic in topics
                if topic.category.casefold()
                == category
            ]

        self.random.shuffle(topics)

        return [
            topic.topic_id
            for topic in topics[:count]
        ]

    # ========================================================
    # RANDOM TOPIC
    # ========================================================

    def random_topic(
        self,
        category: str | None = None,
    ) -> HindiTopic | None:

        topics = list(self.topics.values())

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

        return self.random.choice(topics)

    # ========================================================
    # QUESTION FACTORY
    # ========================================================

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
        options: list[str] | None = None,
    ) -> HindiQuestion:

        self.question_counter += 1

        return HindiQuestion(
            question_id=(
                f"HIN-{self.question_counter:06d}"
            ),
            topic=topic,
            category=category,
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=explanation,
            hint=hint,
            options=options or [],
        )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalize_topic(
        topic: str,
    ) -> str:

        value = topic.strip().casefold()

        aliases = {

            "संज्ञा": "sangya",
            "सर्वनाम": "sarvanam",
            "विशेषण": "visheshan",
            "क्रिया": "kriya",
            "काल": "kaal",
            "लिंग": "ling",
            "वचन": "vachan",
            "कारक": "karak",
            "संधि": "sandhi",
            "समास": "samas",
            "पर्यायवाची": "paryayvachi",
            "विलोम": "vilom",
            "मुहावरे": "muhavare",
            "अलंकार": "alankar",
            "अपठित गद्यांश": "apathit_gadyansh",
            "लेखन": "lekhan",
            "साहित्य": "sahitya",

            "grammar": "sangya",
            "vocabulary": "paryayvachi",
            "idioms": "muhavare",
            "writing": "lekhan",
            "literature": "sahitya",
        }

        return aliases.get(
            value,
            value.replace(" ", "_"),
        )

    # ========================================================
    # ANSWER NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalize_answer(
        answer: Any,
    ) -> str:

        if answer is None:
            return ""

        text = str(answer).strip().casefold()

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        text = text.strip(
            " ।.!?,:;\"'"
        )

        return text

    # ========================================================
    # SEMANTIC MATCH
    # ========================================================

    @staticmethod
    def _semantic_match(
        actual: str,
        expected: str,
    ) -> bool:

        if not actual or not expected:
            return False

        if actual == expected:
            return True

        actual_words = set(
            re.findall(
                r"[\u0900-\u097F]+",
                actual,
            )
        )

        expected_words = set(
            re.findall(
                r"[\u0900-\u097F]+",
                expected,
            )
        )

        if not actual_words or not expected_words:
            return False

        common = (
            actual_words
            & expected_words
        )

        similarity = (
            len(common)
            / max(
                len(expected_words),
                1,
            )
        )

        return similarity >= 0.8

    # ========================================================
    # TEXT UTILITIES
    # ========================================================

    @staticmethod
    def _word_count(
        text: str,
    ) -> int:

        return len(
            re.findall(
                r"[\u0900-\u097F\w'-]+",
                text,
            )
        )

    @staticmethod
    def _sentence_count(
        text: str,
    ) -> int:

        sentences = re.findall(
            r"[^।.!?]+[।.!?]+",
            text,
        )

        if sentences:
            return len(sentences)

        return 1 if text.strip() else 0


# ============================================================
# FACTORY
# ============================================================


def create_hindi_engine(
    *,
    seed: int | None = None,
) -> HindiEngine:

    return HindiEngine(
        seed=seed
    )


# ============================================================
# EXPORTS
# ============================================================


__all__ = [
    "HindiTopic",
    "HindiQuestion",
    "HindiEngine",
    "create_hindi_engine",
]


