"""
RENIX Education — Social Science Subject Engine

File:
    RENIX/education/subjects/social_science.py

Purpose:
    Provides Social Science learning, practice, revision,
    question generation, answer checking and topic management.

Designed to work with:
    RENIX/education/study_manager.py
    RENIX/education/question_generator.py
    RENIX/education/answer_checker.py
    RENIX/education/progress_tracker.py
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
class SocialScienceTopic:
    topic_id: str
    name: str
    subject_area: str
    description: str
    difficulty: str = "medium"
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "name": self.name,
            "subject_area": self.subject_area,
            "description": self.description,
            "difficulty": self.difficulty,
            "keywords": list(self.keywords),
        }


@dataclass
class SocialScienceQuestion:
    question_id: str
    topic: str
    subject_area: str
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
            "subject_area": self.subject_area,
            "question": self.question,
            "answer": self.answer,
            "options": list(self.options),
            "difficulty": self.difficulty,
            "explanation": self.explanation,
            "hint": self.hint,
            "marks": self.marks,
        }


# ============================================================
# SOCIAL SCIENCE ENGINE
# ============================================================


class SocialScienceEngine:

    def __init__(
        self,
        *,
        seed: int | None = None,
    ) -> None:

        self.random = random.Random(seed)

        self.topics: dict[str, SocialScienceTopic] = {}

        self.question_counter = 0

        self._register_topics()

    # ========================================================
    # TOPIC REGISTRATION
    # ========================================================

    def _register_topics(self) -> None:

        topics = [

            # ------------------------------------------------
            # HISTORY
            # ------------------------------------------------

            SocialScienceTopic(
                "history",
                "History",
                "History",
                "Study of past events, societies and historical change.",
                "medium",
                [
                    "events",
                    "movements",
                    "leaders",
                    "chronology",
                    "causes",
                    "effects",
                ],
            ),

            SocialScienceTopic(
                "nationalism_india",
                "Nationalism in India",
                "History",
                "Indian nationalism and the freedom movement.",
                "hard",
                [
                    "gandhi",
                    "non-cooperation",
                    "civil disobedience",
                    "satyagraha",
                    "independence",
                ],
            ),

            SocialScienceTopic(
                "nationalism_europe",
                "The Rise of Nationalism in Europe",
                "History",
                "Development of nationalism and nation-states in Europe.",
                "hard",
                [
                    "france",
                    "germany",
                    "italy",
                    "nationalism",
                    "unification",
                ],
            ),

            SocialScienceTopic(
                "industrialisation",
                "The Age of Industrialisation",
                "History",
                "Industrialisation and its social and economic effects.",
                "hard",
                [
                    "factories",
                    "workers",
                    "machines",
                    "industry",
                    "production",
                ],
            ),

            SocialScienceTopic(
                "global_world",
                "The Making of a Global World",
                "History",
                "Connections between economies, societies and regions.",
                "hard",
                [
                    "trade",
                    "migration",
                    "globalisation",
                    "markets",
                ],
            ),

            SocialScienceTopic(
                "print_culture",
                "Print Culture and the Modern World",
                "History",
                "Development of printing and its impact on society.",
                "medium",
                [
                    "printing",
                    "books",
                    "newspapers",
                    "literacy",
                    "ideas",
                ],
            ),

            # ------------------------------------------------
            # GEOGRAPHY
            # ------------------------------------------------

            SocialScienceTopic(
                "resources",
                "Resources and Development",
                "Geography",
                "Resources, their classification and sustainable development.",
                "medium",
                [
                    "resources",
                    "development",
                    "sustainability",
                    "soil",
                ],
            ),

            SocialScienceTopic(
                "forest_wildlife",
                "Forest and Wildlife Resources",
                "Geography",
                "Forest resources, biodiversity and conservation.",
                "medium",
                [
                    "forests",
                    "wildlife",
                    "biodiversity",
                    "conservation",
                ],
            ),

            SocialScienceTopic(
                "water_resources",
                "Water Resources",
                "Geography",
                "Water availability, conservation and management.",
                "medium",
                [
                    "water",
                    "dams",
                    "irrigation",
                    "conservation",
                ],
            ),

            SocialScienceTopic(
                "agriculture",
                "Agriculture",
                "Geography",
                "Agricultural systems, crops and farming patterns.",
                "medium",
                [
                    "farming",
                    "crops",
                    "irrigation",
                    "farmers",
                ],
            ),

            SocialScienceTopic(
                "manufacturing",
                "Manufacturing Industries",
                "Geography",
                "Industrial production and major manufacturing sectors.",
                "hard",
                [
                    "industry",
                    "manufacturing",
                    "factories",
                    "pollution",
                ],
            ),

            SocialScienceTopic(
                "lifelines",
                "Lifelines of National Economy",
                "Geography",
                "Transport, communication and trade networks.",
                "medium",
                [
                    "transport",
                    "communication",
                    "trade",
                    "roads",
                    "railways",
                ],
            ),

            # ------------------------------------------------
            # POLITICAL SCIENCE
            # ------------------------------------------------

            SocialScienceTopic(
                "power_sharing",
                "Power Sharing",
                "Political Science",
                "Forms and importance of sharing political power.",
                "medium",
                [
                    "democracy",
                    "power",
                    "belgium",
                    "srilanka",
                ],
            ),

            SocialScienceTopic(
                "federalism",
                "Federalism",
                "Political Science",
                "Distribution of power between different levels of government.",
                "medium",
                [
                    "union",
                    "state",
                    "local",
                    "government",
                ],
            ),

            SocialScienceTopic(
                "gender_religion_caste",
                "Gender, Religion and Caste",
                "Political Science",
                "Social differences and their relationship with politics.",
                "hard",
                [
                    "gender",
                    "religion",
                    "caste",
                    "equality",
                ],
            ),

            SocialScienceTopic(
                "political_parties",
                "Political Parties",
                "Political Science",
                "Political parties, their functions and challenges.",
                "medium",
                [
                    "parties",
                    "elections",
                    "leadership",
                    "democracy",
                ],
            ),

            SocialScienceTopic(
                "outcomes_democracy",
                "Outcomes of Democracy",
                "Political Science",
                "How democracy performs and what outcomes it produces.",
                "hard",
                [
                    "democracy",
                    "accountability",
                    "development",
                    "equality",
                ],
            ),

            SocialScienceTopic(
                "challenges_democracy",
                "Challenges to Democracy",
                "Political Science",
                "Major challenges faced by democratic systems.",
                "hard",
                [
                    "democracy",
                    "reforms",
                    "institutions",
                    "participation",
                ],
            ),

            # ------------------------------------------------
            # ECONOMICS
            # ------------------------------------------------

            SocialScienceTopic(
                "development",
                "Development",
                "Economics",
                "Different ideas of development and development indicators.",
                "medium",
                [
                    "income",
                    "health",
                    "education",
                    "development",
                ],
            ),

            SocialScienceTopic(
                "sectors",
                "Sectors of the Indian Economy",
                "Economics",
                "Primary, secondary and tertiary sectors of the economy.",
                "medium",
                [
                    "primary",
                    "secondary",
                    "tertiary",
                    "employment",
                ],
            ),

            SocialScienceTopic(
                "money_credit",
                "Money and Credit",
                "Economics",
                "Money, banks, loans and formal and informal credit.",
                "medium",
                [
                    "money",
                    "banks",
                    "credit",
                    "loans",
                ],
            ),

            SocialScienceTopic(
                "globalisation",
                "Globalisation and the Indian Economy",
                "Economics",
                "Globalisation and its impact on the Indian economy.",
                "hard",
                [
                    "globalisation",
                    "MNC",
                    "trade",
                    "investment",
                ],
            ),

            SocialScienceTopic(
                "consumer_rights",
                "Consumer Rights",
                "Economics",
                "Consumer awareness, protection and rights.",
                "medium",
                [
                    "consumer",
                    "rights",
                    "markets",
                    "protection",
                ],
            ),
        ]

        for topic in topics:
            self.register_topic(topic)

    def register_topic(
        self,
        topic: SocialScienceTopic,
    ) -> None:

        self.topics[
            self._normalize_topic(topic.topic_id)
        ] = topic

    # ========================================================
    # TOPIC ACCESS
    # ========================================================

    def get_topics(
        self,
        subject_area: str | None = None,
    ) -> list[dict[str, Any]]:

        topics = list(self.topics.values())

        if subject_area:

            value = subject_area.casefold()

            topics = [
                topic
                for topic in topics
                if topic.subject_area.casefold()
                == value
            ]

        return [
            topic.to_dict()
            for topic in topics
        ]

    def get_topic(
        self,
        topic: str,
    ) -> SocialScienceTopic | None:

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

            searchable = " ".join(
                [
                    topic.topic_id,
                    topic.name,
                    topic.subject_area,
                    topic.description,
                    *topic.keywords,
                ]
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
    ) -> SocialScienceQuestion:

        topic_id = self._normalize_topic(topic)

        generators = {

            "nationalism_india":
                self._nationalism_india_question,

            "nationalism_europe":
                self._nationalism_europe_question,

            "industrialisation":
                self._industrialisation_question,

            "global_world":
                self._global_world_question,

            "print_culture":
                self._print_culture_question,

            "resources":
                self._resources_question,

            "forest_wildlife":
                self._forest_question,

            "water_resources":
                self._water_question,

            "agriculture":
                self._agriculture_question,

            "manufacturing":
                self._manufacturing_question,

            "lifelines":
                self._lifelines_question,

            "power_sharing":
                self._power_sharing_question,

            "federalism":
                self._federalism_question,

            "gender_religion_caste":
                self._gender_religion_caste_question,

            "political_parties":
                self._political_parties_question,

            "outcomes_democracy":
                self._outcomes_democracy_question,

            "challenges_democracy":
                self._challenges_democracy_question,

            "development":
                self._development_question,

            "sectors":
                self._sectors_question,

            "money_credit":
                self._money_credit_question,

            "globalisation":
                self._globalisation_question,

            "consumer_rights":
                self._consumer_rights_question,

            "history":
                self._generic_history_question,
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
    ) -> list[SocialScienceQuestion]:

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
    # HISTORY QUESTIONS
    # ========================================================

    def _generic_history_question(
        self,
        difficulty: str,
    ) -> SocialScienceQuestion:

        questions = [
            (
                "इतिहास के अध्ययन से हमें क्या समझने में सहायता मिलती है?",
                "अतीत की घटनाओं, समाजों और परिवर्तनों को समझने में।",
            ),
            (
                "ऐतिहासिक घटनाओं के कारणों का अध्ययन क्यों महत्वपूर्ण है?",
                "इससे यह समझने में सहायता मिलती है कि कोई घटना क्यों हुई और उसके क्या परिणाम हुए।",
            ),
        ]

        question, answer = self.random.choice(
            questions
        )

        return self._question(
            topic="history",
            subject_area="History",
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=answer,
            hint="अतीत की घटनाओं और उनके प्रभाव पर ध्यान दें।",
            marks=2,
        )

    def _nationalism_india_question(
        self,
        difficulty: str,
    ) -> SocialScienceQuestion:

        questions = [
            (
                "महात्मा गांधी ने भारत के राष्ट्रीय आंदोलन में सत्याग्रह का प्रयोग क्यों किया?",
                "अन्याय के विरुद्ध अहिंसक प्रतिरोध और सत्य के आधार पर संघर्ष के लिए।",
                "सत्याग्रह का आधार सत्य और अहिंसा था।",
                "राष्ट्रीय आंदोलन में अहिंसक संघर्ष पर ध्यान दें।",
            ),
            (
                "असहयोग आंदोलन का मुख्य उद्देश्य क्या था?",
                "ब्रिटिश शासन के साथ सहयोग समाप्त करके औपनिवेशिक शासन का विरोध करना।",
                "आंदोलन में सरकारी संस्थाओं और विदेशी वस्तुओं के बहिष्कार जैसे उपाय शामिल थे।",
                "नाम में ही 'असहयोग' का अर्थ छिपा है।",
            ),
            (
                "सविनय अवज्ञा आंदोलन किस विचार पर आधारित था?",
                "अन्यायपूर्ण कानूनों का अहिंसक ढंग से उल्लंघन करके उनका विरोध करना।",
                "सविनय अवज्ञा में अन्यायपूर्ण कानूनों का शांतिपूर्ण विरोध किया गया।",
                "कानूनों के अहिंसक विरोध को याद रखें।",
            ),
        ]

        return self._choice_question(
            "nationalism_india",
            "History",
            questions,
            difficulty,
            marks=2,
        )

    def _nationalism_europe_question(
        self,
        difficulty: str,
    ) -> SocialScienceQuestion:

        questions = [
            (
                "यूरोप में राष्ट्रवाद के विकास में फ्रांसीसी क्रांति का क्या महत्व था?",
                "इसने स्वतंत्रता, समानता और राष्ट्र की अवधारणाओं को मजबूत किया।",
                "फ्रांसीसी क्रांति के विचारों ने यूरोप में आधुनिक राष्ट्रवाद को प्रभावित किया।",
                "स्वतंत्रता और समानता जैसे क्रांतिकारी विचारों को याद रखें।",
            ),
            (
                "जर्मनी के एकीकरण में बिस्मार्क की क्या भूमिका थी?",
                "उन्होंने प्रशा के नेतृत्व में जर्मन राज्यों के एकीकरण में महत्वपूर्ण भूमिका निभाई।",
                "बिस्मार्क ने युद्ध और कूटनीति का उपयोग करके एकीकरण की प्रक्रिया को आगे बढ़ाया।",
                "प्रशा और जर्मन एकीकरण को जोड़कर देखें।",
            ),
        ]

        return self._choice_question(
            "nationalism_europe",
            "History",
            questions,
            difficulty,
            marks=3,
        )

    def _industrialisation_question(
        self,
        difficulty: str,
    ) -> SocialScienceQuestion:

        questions = [
            (
                "औद्योगीकरण ने श्रमिकों के जीवन को किस प्रकार प्रभावित किया?",
                "कारखानों में रोजगार के अवसर बढ़े, लेकिन लंबे काम के घंटे, कम मजदूरी और कठिन परिस्थितियाँ भी थीं।",
                "औद्योगीकरण के लाभ और श्रमिकों की कठिनाइयों दोनों का उल्लेख करना चाहिए।",
                "रोजगार और काम की परिस्थितियों दोनों पर विचार करें।",
            ),
            (
                "मशीनों के बढ़ते प्रयोग का उत्पादन पर क्या प्रभाव पड़ा?",
                "उत्पादन की गति और मात्रा बढ़ी तथा उत्पादन की प्रक्रिया में परिवर्तन आया।",
                "मशीनीकरण ने बड़े पैमाने पर उत्पादन को बढ़ावा दिया।",
                "मशीन और उत्पादन के संबंध पर ध्यान दें।",
            ),
        ]

        return self._choice_question(
            "industrialisation",
            "History",
            questions,
            difficulty,
            marks=3,
        )

    def _global_world_question(
        self,
        difficulty: str,
    ) -> SocialScienceQuestion:

        questions = [
            (
                "वैश्विक व्यापार के विकास में परिवहन का क्या महत्व रहा है?",
                "बेहतर परिवहन ने वस्तुओं, लोगों और विचारों की दूर-दूर तक आवाजाही को आसान बनाया।",
                "परिवहन ने विभिन्न क्षेत्रों को आर्थिक रूप से जोड़ने में सहायता की।",
                "लोगों और वस्तुओं की आवाजाही के बारे में सोचें।",
            ),
            (
                "वैश्वीकरण से क्या तात्पर्य है?",
                "दुनिया की अर्थव्यवस्थाओं और समाजों के बीच बढ़ते संपर्क और परस्पर निर्भरता से।",
                "वैश्वीकरण में व्यापार, निवेश, तकनीक और संचार की महत्वपूर्ण भूमिका है।",
                "देशों के बीच बढ़ते संबंधों पर ध्यान दें।",
            ),
        ]

        return self._choice_question(
            "global_world",
            "History",
            questions,
            difficulty,
            marks=3,
        )

    def _print_culture_question(
        self,
        difficulty: str,
    ) -> SocialScienceQuestion:

        questions = [
            (
                "मुद्रण संस्कृति ने विचारों के प्रसार को कैसे प्रभावित किया?",
                "मुद्रित पुस्तकों, समाचारपत्रों और अन्य सामग्री के माध्यम से विचार अधिक लोगों तक तेजी से पहुँचे।",
                "मुद्रण ने ज्ञान और विचारों के प्रसार को व्यापक बनाया।",
                "किताबों और समाचारपत्रों के प्रसार को याद करें।",
            ),
            (
                "मुद्रण के विकास का शिक्षा पर क्या प्रभाव पड़ा?",
                "पुस्तकों की उपलब्धता बढ़ने से पढ़ने और सीखने के अवसर बढ़े।",
                "मुद्रित सामग्री ने शिक्षा और साक्षरता के विस्तार में सहायता की।",
                "पुस्तकों की उपलब्धता के बारे में सोचें।",
            ),
        ]

        return self._choice_question(
            "print_culture",
            "History",
            questions,
            difficulty,
            marks=3,
        )

    # ========================================================
    # GEOGRAPHY QUESTIONS
    # ========================================================

    def _resources_question(
        self,
        difficulty: str,
    ) -> SocialScienceQuestion:

        questions = [
            (
                "संसाधन संरक्षण क्यों आवश्यक है?",
                "भविष्य की पीढ़ियों के लिए संसाधनों की उपलब्धता बनाए रखने के लिए।",
                "संसाधनों का सीमित और विवेकपूर्ण उपयोग सतत विकास के लिए आवश्यक है।",
                "भविष्य की पीढ़ियों को ध्यान में रखें।",
            ),
            (
                "सतत विकास का क्या अर्थ है?",
                "ऐसा विकास जिसमें वर्तमान की आवश्यकताएँ पूरी हों और भविष्य की पीढ़ियों की आवश्यकताओं से समझौता न हो।",
                "सतत विकास वर्तमान और भविष्य दोनों की आवश्यकताओं को संतुलित करता है।",
                "वर्तमान और भविष्य के बीच संतुलन सोचें।",
            ),
        ]

        return self._choice_question(
            "resources",
            "Geography",
            questions,
            difficulty,
            marks=3,
        )

    def _forest_question(
        self,
        difficulty: str,
    ) -> SocialScienceQuestion:

        questions = [
            (
                "जैव विविधता का संरक्षण क्यों आवश्यक है?",
                "पारिस्थितिक संतुलन बनाए रखने और विभिन्न प्रजातियों के अस्तित्व की रक्षा के लिए।",
                "जैव विविधता पारिस्थितिकी तंत्र के संतुलन में महत्वपूर्ण भूमिका निभाती है।",
                "प्रजातियों और पारिस्थितिक संतुलन पर ध्यान दें।",
            ),
            (
                "वनों का संरक्षण क्यों महत्वपूर्ण है?",
                "वन पर्यावरणीय संतुलन, जैव विविधता, मिट्टी और जल संरक्षण में महत्वपूर्ण भूमिका निभाते हैं।",
                "वन अनेक जीवों का आवास भी हैं।",
                "वनों के पर्यावरणीय कार्यों को याद करें।",
            ),
        ]

        return self._choice_question(
            "forest_wildlife",
            "Geography",
            questions,
            difficulty,
            marks=3,
        )

    def _water_question(
        self,
        difficulty: str,
    ) -> SocialScienceQuestion:

        questions = [
            (
                "वर्षा जल संचयन क्या है?",
                "वर्षा के जल को एकत्र करके उसका उपयोग या भूजल पुनर्भरण करना।",
                "वर्षा जल संचयन जल संरक्षण का महत्वपूर्ण उपाय है।",
                "वर्षा के पानी को बचाने की प्रक्रिया सोचें।",
            ),
            (
                "बहुउद्देशीय नदी घाटी परियोजनाओं के दो लाभ लिखिए।",
                "सिंचाई और जलविद्युत उत्पादन जैसे लाभ।",
                "इन परियोजनाओं से बाढ़ नियंत्रण और जल आपूर्ति जैसे अन्य लाभ भी हो सकते हैं।",
                "सिंचाई और बिजली को याद रखें।",
            ),
        ]

        return self._choice_question(
            "water_resources",
            "Geography",
            questions,
            difficulty,
            marks=3,
        )

    def _agriculture_question(
        self,
        difficulty: str,
    ) -> SocialScienceQuestion:

        questions = [
            (
                "भारत में कृषि का महत्व क्यों है?",
                "यह रोजगार, खाद्य आपूर्ति और उद्योगों के लिए कच्चा माल प्रदान करती है।",
                "कृषि भारतीय अर्थव्यवस्था और ग्रामीण जीवन का महत्वपूर्ण हिस्सा है।",
                "खाद्य, रोजगार और उद्योग तीनों पर विचार करें।",
            ),
            (
                "रबी फसलें सामान्यतः कब बोई जाती हैं?",
                "सर्दियों के मौसम में।",
                "रबी फसलें सामान्यतः ठंडे मौसम में बोई जाती हैं और गर्मियों के आसपास काटी जाती हैं।",
                "खरीफ के विपरीत मौसम याद करें।",
            ),
        ]

        return self._choice_question(
            "agriculture",
            "Geography",
            questions,
            difficulty,
            marks=2,
        )

    def _manufacturing_question(
        self,
        difficulty: str,
    ) -> SocialScienceQuestion:

        questions = [
            (
                "विनिर्माण उद्योगों का अर्थव्यवस्था में क्या महत्व है?",
                "वे रोजगार, उत्पादन, निर्यात और आर्थिक विकास में योगदान करते हैं।",
                "विनिर्माण कृषि और अन्य क्षेत्रों से प्राप्त कच्चे माल को उपयोगी वस्तुओं में बदलता है।",
                "रोजगार और उत्पादन पर ध्यान दें।",
            ),
            (
                "औद्योगिक प्रदूषण को कम करने के लिए एक उपाय बताइए।",
                "अपशिष्ट का उपचार करके और स्वच्छ तकनीक का प्रयोग करके प्रदूषण कम किया जा सकता है।",
                "उद्योगों को प्रदूषण नियंत्रण तकनीकों का प्रयोग करना चाहिए।",
                "उत्पादन से निकलने वाले अपशिष्ट के बारे में सोचें।",
            ),
        ]

        return self._choice_question(
            "manufacturing",
            "Geography",
            questions,
            difficulty,
            marks=3,
        )

    def _lifelines_question(
        self,
        difficulty: str,
    ) -> SocialScienceQuestion:

        questions = [
            (
                "परिवहन को राष्ट्रीय अर्थव्यवस्था की जीवन रेखा क्यों कहा जाता है?",
                "क्योंकि यह लोगों, वस्तुओं और सेवाओं की आवाजाही को संभव बनाता है और विभिन्न क्षेत्रों को जोड़ता है।",
                "परिवहन आर्थिक गतिविधियों और व्यापार के लिए आवश्यक है।",
                "लोगों और वस्तुओं की आवाजाही पर ध्यान दें।",
            ),
            (
                "संचार का आर्थिक विकास में क्या महत्व है?",
                "यह सूचना के तेज आदान-प्रदान को संभव बनाता है और व्यापार तथा सेवाओं को सहायता देता है।",
                "आधुनिक अर्थव्यवस्था में सूचना का तेज संचार महत्वपूर्ण है।",
                "सूचना के आदान-प्रदान को याद रखें।",
            ),
        ]

        return self._choice_question(
            "lifelines",
            "Geography",
            questions,
            difficulty,
            marks=3,
        )

    # ========================================================
    # POLITICAL SCIENCE QUESTIONS
    # ========================================================

    def _power_sharing_question(
        self,
        difficulty: str,
    ) -> SocialScienceQuestion:

        questions = [
            (
                "लोकतंत्र में सत्ता की साझेदारी क्यों आवश्यक है?",
                "यह सामाजिक संघर्ष को कम करती है और विभिन्न समूहों को शासन में भागीदारी देती है।",
                "सत्ता की साझेदारी लोकतंत्र की स्थिरता और समावेशिता को मजबूत कर सकती है।",
                "संघर्ष और भागीदारी दोनों पर विचार करें।",
            ),
            (
                "सत्ता की साझेदारी का एक prudential reason क्या है?",
                "यह सामाजिक संघर्ष और राजनीतिक अस्थिरता की संभावना को कम करती है।",
                "सत्ता बाँटने से विभिन्न समूहों के बीच तनाव कम हो सकता है।",
                "संघर्ष कम होने पर ध्यान दें।",
            ),
        ]

        return self._choice_question(
            "power_sharing",
            "Political Science",
            questions,
            difficulty,
            marks=3,
        )

    def _federalism_question(
        self,
        difficulty: str,
    ) -> SocialScienceQuestion:

        questions = [
            (
                "संघवाद क्या है?",
                "सरकार के विभिन्न स्तरों के बीच संवैधानिक रूप से शक्तियों का विभाजन संघवाद कहलाता है।",
                "संघवाद में केंद्र और राज्य जैसे विभिन्न स्तरों की सरकारें होती हैं।",
                "शक्तियों के विभाजन को याद रखें।",
            ),
            (
                "भारत में सरकार के तीन प्रमुख स्तर कौन-से हैं?",
                "केंद्र सरकार, राज्य सरकार और स्थानीय सरकार।",
                "भारत में संघ, राज्य और स्थानीय स्तर पर शासन की व्यवस्था है।",
                "केंद्र से स्थानीय स्तर तक सोचें।",
            ),
        ]

        return self._choice_question(
            "federalism",
            "Political Science",
            questions,
            difficulty,
            marks=2,
        )

    def _gender_religion_caste_question(
        self,
        difficulty: str,
    ) -> SocialScienceQuestion:

        questions = [
            (
                "राजनीति में लैंगिक समानता क्यों महत्वपूर्ण है?",
                "क्योंकि लोकतंत्र में सभी नागरिकों को समान राजनीतिक अवसर और अधिकार मिलने चाहिए।",
                "समान भागीदारी लोकतांत्रिक व्यवस्था को अधिक प्रतिनिधिक बनाती है।",
                "समान अधिकार और भागीदारी पर ध्यान दें।",
            ),
            (
                "जाति व्यवस्था राजनीति को किस प्रकार प्रभावित कर सकती है?",
                "जातीय पहचान चुनावी राजनीति, प्रतिनिधित्व और सामाजिक समूहों की मांगों को प्रभावित कर सकती है।",
                "जाति और राजनीति का संबंध विभिन्न परिस्थितियों में अलग-अलग रूप ले सकता है।",
                "पहचान और प्रतिनिधित्व पर विचार करें।",
            ),
        ]

        return self._choice_question(
            "gender_religion_caste",
            "Political Science",
            questions,
            difficulty,
            marks=3,
        )

    def _political_parties_question(
        self,
        difficulty: str,
    ) -> SocialScienceQuestion:

        questions = [
            (
                "राजनीतिक दल का एक प्रमुख कार्य क्या है?",
                "चुनाव लड़ना, नीतियाँ प्रस्तुत करना और सरकार बनाने में भूमिका निभाना।",
                "राजनीतिक दल जनता और सरकार के बीच महत्वपूर्ण कड़ी होते हैं।",
                "चुनाव और सरकार के गठन के बारे में सोचें।",
            ),
            (
                "राजनीतिक दल लोकतंत्र में क्यों आवश्यक हैं?",
                "वे विभिन्न नीतिगत विकल्प प्रस्तुत करते हैं और जनता को राजनीतिक विकल्प देते हैं।",
                "दलों के बिना बड़े पैमाने पर लोकतांत्रिक प्रतिनिधित्व कठिन हो सकता है।",
                "नीतियों और प्रतिनिधित्व पर ध्यान दें।",
            ),
        ]

        return self._choice_question(
            "political_parties",
            "Political Science",
            questions,
            difficulty,
            marks=3,
        )

    def _outcomes_democracy_question(
        self,
        difficulty: str,
    ) -> SocialScienceQuestion:

        questions = [
            (
                "लोकतंत्र में जवाबदेही का क्या महत्व है?",
                "सरकार को अपने निर्णयों और कार्यों के लिए जनता के प्रति उत्तरदायी बनाना।",
                "जवाबदेही नागरिकों को सरकार के कार्यों पर प्रश्न उठाने का अवसर देती है।",
                "सरकार और जनता के संबंध पर ध्यान दें।",
            ),
            (
                "लोकतंत्र निर्णय लेने की गुणवत्ता को कैसे प्रभावित कर सकता है?",
                "चर्चा, विचार-विमर्श और विभिन्न हितों की भागीदारी के कारण निर्णय अधिक विचारपूर्ण हो सकते हैं।",
                "लोकतंत्र में निर्णय प्रक्रिया में भागीदारी और चर्चा महत्वपूर्ण होती है।",
                "चर्चा और भागीदारी को याद रखें।",
            ),
        ]

        return self._choice_question(
            "outcomes_democracy",
            "Political Science",
            questions,
            difficulty,
            marks=3,
        )

    def _challenges_democracy_question(
        self,
        difficulty: str,
    ) -> SocialScienceQuestion:

        questions = [
            (
                "लोकतंत्र के सामने कोई एक प्रमुख चुनौती बताइए।",
                "लोकतांत्रिक संस्थाओं को अधिक जवाबदेह, सहभागी और प्रभावी बनाना एक प्रमुख चुनौती है।",
                "लोकतंत्र को मजबूत करने के लिए संस्थागत और नागरिक भागीदारी दोनों आवश्यक हैं।",
                "लोकतांत्रिक संस्थाओं की गुणवत्ता पर विचार करें।",
            ),
        ]

        return self._choice_question(
            "challenges_democracy",
            "Political Science",
            questions,
            difficulty,
            marks=2,
        )

    # ========================================================
    # ECONOMICS QUESTIONS
    # ========================================================

    def _development_question(
        self,
        difficulty: str,
    ) -> SocialScienceQuestion:

        questions = [
            (
                "लोग विकास को अलग-अलग तरीके से क्यों देखते हैं?",
                "क्योंकि अलग-अलग लोगों की आवश्यकताएँ, परिस्थितियाँ और प्राथमिकताएँ अलग होती हैं।",
                "विकास केवल आय से नहीं बल्कि स्वास्थ्य, शिक्षा, सुरक्षा और अन्य पहलुओं से भी जुड़ा है।",
                "लोगों की आवश्यकताओं में अंतर सोचें।",
            ),
            (
                "प्रति व्यक्ति आय क्या दर्शाती है?",
                "किसी देश या क्षेत्र की कुल आय को उसकी जनसंख्या से विभाजित करने पर प्राप्त औसत आय।",
                "यह औसत आय का एक माप है, लेकिन अकेले विकास की पूरी तस्वीर नहीं देता।",
                "कुल आय और जनसंख्या के संबंध को याद रखें।",
            ),
        ]

        return self._choice_question(
            "development",
            "Economics",
            questions,
            difficulty,
            marks=3,
        )

    def _sectors_question(
        self,
        difficulty: str,
    ) -> SocialScienceQuestion:

        questions = [
            (
                "प्राथमिक क्षेत्र क्या है?",
                "वह क्षेत्र जिसमें प्राकृतिक संसाधनों से सीधे उत्पादन किया जाता है, जैसे कृषि और मछली पालन।",
                "प्राथमिक क्षेत्र प्राकृतिक संसाधनों पर सीधे निर्भर करता है।",
                "कृषि और प्राकृतिक संसाधनों को याद करें।",
            ),
            (
                "तृतीयक क्षेत्र को सेवा क्षेत्र क्यों कहा जाता है?",
                "क्योंकि यह वस्तुओं के बजाय विभिन्न सेवाएँ प्रदान करता है।",
                "बैंकिंग, परिवहन और शिक्षा जैसी सेवाएँ तृतीयक क्षेत्र में आती हैं।",
                "सेवाओं के उदाहरण सोचें।",
            ),
        ]

        return self._choice_question(
            "sectors",
            "Economics",
            questions,
            difficulty,
            marks=3,
        )

    def _money_credit_question(
        self,
        difficulty: str,
    ) -> SocialScienceQuestion:

        questions = [
            (
                "बैंकों का एक प्रमुख कार्य क्या है?",
                "लोगों से जमा स्वीकार करना और जरूरतमंदों को ऋण देना।",
                "बैंक बचत को ऋण के रूप में उपलब्ध कराने में महत्वपूर्ण भूमिका निभाते हैं।",
                "जमा और ऋण दोनों को याद रखें।",
            ),
            (
                "संपार्श्विक क्या है?",
                "ऋण लेते समय ऋणदाता के पास सुरक्षा के रूप में रखी गई संपत्ति।",
                "यदि ऋण चुकाया नहीं जाता तो संपार्श्विक ऋणदाता के लिए सुरक्षा का काम करता है।",
                "ऋण की सुरक्षा के बारे में सोचें।",
            ),
        ]

        return self._choice_question(
            "money_credit",
            "Economics",
            questions,
            difficulty,
            marks=3,
        )

    def _globalisation_question(
        self,
        difficulty: str,
    ) -> SocialScienceQuestion:

        questions = [
            (
                "बहुराष्ट्रीय कंपनी (MNC) क्या है?",
                "ऐसी कंपनी जो एक से अधिक देशों में उत्पादन या व्यापार की गतिविधियाँ संचालित करती है।",
                "MNCs विभिन्न देशों में निवेश और उत्पादन नेटवर्क स्थापित कर सकती हैं।",
                "एक से अधिक देशों में काम करने वाली कंपनी सोचें।",
            ),
            (
                "वैश्वीकरण को बढ़ावा देने वाला एक प्रमुख कारक क्या है?",
                "तकनीकी प्रगति और परिवहन तथा संचार में सुधार।",
                "बेहतर संचार और परिवहन ने देशों को आर्थिक रूप से अधिक जोड़ा है।",
                "तकनीक और संपर्क पर ध्यान दें।",
            ),
        ]

        return self._choice_question(
            "globalisation",
            "Economics",
            questions,
            difficulty,
            marks=3,
        )

    def _consumer_rights_question(
        self,
        difficulty: str,
    ) -> SocialScienceQuestion:

        questions = [
            (
                "उपभोक्ता जागरूकता क्यों आवश्यक है?",
                "ताकि उपभोक्ता सही वस्तु चुन सकें और शोषण से अपने अधिकारों की रक्षा कर सकें।",
                "जानकारी होने से उपभोक्ता बेहतर निर्णय ले सकते हैं।",
                "अधिकार और सही चुनाव पर ध्यान दें।",
            ),
            (
                "उपभोक्ता का सूचना का अधिकार क्या है?",
                "उपभोक्ता को वस्तु या सेवा की गुणवत्ता, मात्रा, कीमत और अन्य आवश्यक जानकारी प्राप्त करने का अधिकार है।",
                "जानकारी उपभोक्ता को सूचित निर्णय लेने में सहायता करती है।",
                "वस्तु की कीमत और गुणवत्ता की जानकारी याद रखें।",
            ),
        ]

        return self._choice_question(
            "consumer_rights",
            "Economics",
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
        subject_area: str,
        questions: list[tuple[str, str, str, str]],
        difficulty: str,
        *,
        marks: int,
    ) -> SocialScienceQuestion:

        question, answer, explanation, hint = (
            self.random.choice(questions)
        )

        return self._question(
            topic=topic,
            subject_area=subject_area,
            question=question,
            answer=answer,
            difficulty=difficulty,
            explanation=explanation,
            hint=hint,
            marks=marks,
        )

    def _generic_question(
        self,
        topic: str,
        difficulty: str,
    ) -> SocialScienceQuestion:

        data = self.get_topic(topic)

        if data is None:

            return self._question(
                topic=topic,
                subject_area="Social Science",
                question=(
                    f"Explain the Social Science topic "
                    f"'{topic.replace('_', ' ')}'."
                ),
                answer="",
                difficulty=difficulty,
                explanation=(
                    "Define the topic, explain its main features "
                    "and provide relevant examples."
                ),
                hint=(
                    "Start with a definition and then explain "
                    "the important points."
                ),
                marks=3,
            )

        return self._question(
            topic=topic,
            subject_area=data.subject_area,
            question=(
                f"Explain the main idea of "
                f"'{data.name}'."
            ),
            answer=data.description,
            difficulty=difficulty,
            explanation=data.description,
            hint=(
                "Begin with the definition and then explain "
                "its importance."
            ),
            marks=3,
        )

    def _question(
        self,
        *,
        topic: str,
        subject_area: str,
        question: str,
        answer: Any,
        difficulty: str,
        explanation: str,
        hint: str,
        marks: int,
        options: list[str] | None = None,
    ) -> SocialScienceQuestion:

        self.question_counter += 1

        return SocialScienceQuestion(
            question_id=(
                f"SST-{self.question_counter:06d}"
            ),
            topic=topic,
            subject_area=subject_area,
            question=question,
            answer=answer,
            options=options or [],
            difficulty=difficulty,
            explanation=explanation,
            hint=hint,
            marks=marks,
        )

    # ========================================================
    # ANSWER CHECKING
    # ========================================================

    def check_answer(
        self,
        question: SocialScienceQuestion,
        user_answer: str,
    ) -> dict[str, Any]:

        actual = self._normalize_text(
            user_answer
        )

        expected = self._normalize_text(
            question.answer
        )

        if not actual:

            return {
                "correct": False,
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
                "score": question.marks,
                "max_score": question.marks,
                "accuracy": 100.0,
                "question_id": question.question_id,
                "correct_answer": question.answer,
                "user_answer": user_answer,
                "feedback": "बहुत बढ़िया! उत्तर सही है।",
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
                    "उत्तर का मुख्य विचार सही है, "
                    "लेकिन इसे और सटीक बनाया जा सकता है।"
                ),
                "explanation": question.explanation,
            }

        if similarity >= 0.40:

            score = max(
                0,
                int(
                    question.marks
                    * similarity
                ),
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
                    "कुछ विचार सही दिशा में हैं। "
                    "मुख्य बिंदु जोड़ें।"
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
            "feedback": "उत्तर सही नहीं है। समाधान देखें और दोबारा प्रयास करें।",
            "hint": question.hint,
            "explanation": question.explanation,
        }

    # ========================================================
    # TEXT SIMILARITY
    # ========================================================

    @staticmethod
    def _normalize_text(
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
    # HINTS AND SOLUTIONS
    # ========================================================

    def get_hint(
        self,
        question: SocialScienceQuestion,
    ) -> str:

        return question.hint

    def get_solution(
        self,
        question: SocialScienceQuestion,
    ) -> str:

        return question.explanation

    # ========================================================
    # TOPIC RECOMMENDATION
    # ========================================================

    def recommend_topics(
        self,
        *,
        subject_area: str | None = None,
        weak_topics: Iterable[str] | None = None,
        count: int = 5,
    ) -> list[str]:

        if weak_topics:

            return list(
                weak_topics
            )[:count]

        topics = list(
            self.topics.values()
        )

        if subject_area:

            area = subject_area.casefold()

            topics = [
                topic
                for topic in topics
                if topic.subject_area.casefold()
                == area
            ]

        self.random.shuffle(topics)

        return [
            topic.topic_id
            for topic in topics[:count]
        ]

    # ========================================================
    # RANDOM QUESTION
    # ========================================================

    def generate_random_question(
        self,
        *,
        subject_area: str | None = None,
        difficulty: str = "medium",
    ) -> SocialScienceQuestion:

        topic = self.random_topic(
            subject_area=subject_area
        )

        if topic is None:
            return self._generic_question(
                "social_science",
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
        subject_area: str | None = None,
    ) -> SocialScienceTopic | None:

        topics = list(
            self.topics.values()
        )

        if subject_area:

            area = subject_area.casefold()

            topics = [
                topic
                for topic in topics
                if topic.subject_area.casefold()
                == area
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
        subject_area: str | None = None,
        count: int = 10,
        difficulty: str = "medium",
    ) -> list[SocialScienceQuestion]:

        questions = []

        used_topics: set[str] = set()

        available_topics = list(
            self.topics.values()
        )

        if subject_area:

            area = subject_area.casefold()

            available_topics = [
                topic
                for topic in available_topics
                if topic.subject_area.casefold()
                == area
            ]

        self.random.shuffle(
            available_topics
        )

        for topic in available_topics:

            if len(questions) >= count:
                break

            question = self.generate_question(
                topic.topic_id,
                difficulty=difficulty,
            )

            questions.append(question)

            used_topics.add(
                topic.topic_id
            )

        while len(questions) < count:

            topic = self.random_topic(
                subject_area=subject_area
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
    # MARKS / DIFFICULTY HELPERS
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

        text = question.casefold()

        hard_words = [
            "विश्लेषण",
            "विश्लेषित",
            "तुलना",
            "मूल्यांकन",
            "औचित्य",
            "व्याख्या",
            "प्रभाव",
            "कारण एवं परिणाम",
            "चर्चा",
            "महत्वपूर्ण",
            "समझाइए",
            "विस्तार से",
        ]

        easy_words = [
            "क्या है",
            "कौन",
            "कब",
            "कहाँ",
            "नाम बताइए",
            "परिभाषा",
            "एक कारण",
            "एक उदाहरण",
        ]

        if any(
            word in text
            for word in hard_words
        ):
            return "hard"

        if any(
            word in text
            for word in easy_words
        ):
            return "easy"

        return "medium"

    # ========================================================
    # SEARCH / FILTER
    # ========================================================

    def topics_by_difficulty(
        self,
        difficulty: str,
    ) -> list[SocialScienceTopic]:

        difficulty = difficulty.casefold()

        return [
            topic
            for topic in self.topics.values()
            if topic.difficulty.casefold()
            == difficulty
        ]

    def topics_by_area(
        self,
        subject_area: str,
    ) -> list[SocialScienceTopic]:

        area = subject_area.casefold()

        return [
            topic
            for topic in self.topics.values()
            if topic.subject_area.casefold()
            == area
        ]


# ============================================================
# FACTORY
# ============================================================


def create_social_science_engine(
    *,
    seed: int | None = None,
) -> SocialScienceEngine:

    return SocialScienceEngine(
        seed=seed
    )


# ============================================================
# MODULE EXPORTS
# ============================================================


__all__ = [
    "SocialScienceTopic",
    "SocialScienceQuestion",
    "SocialScienceEngine",
    "create_social_science_engine",
]


