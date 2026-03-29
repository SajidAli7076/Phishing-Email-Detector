import pandas as pd
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# ============================================================
# 📧 EMAIL SAFETY CHECKER
# ============================================================

class PhishingDetector:
    def __init__(self):
        # This part helps the computer turn words into numbers it can understand
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")

        # This is the "brain" that learns to tell the difference between good and bad emails
        self.model = MultinomialNB()

        # These are "red flag" words that usually appear in scam emails
        self.risk_keywords = [
            "bank", "verify", "urgent", "password", "login",
            "suspended", "win", "prize", "money", "claim",
            "security", "action", "alert", "official"
        ]
        self._train_engine()

    def _clean_text(self, text):
        # Make everything lowercase so 'BANK' and 'bank' look the same
        text = text.lower()
        # Find website links and replace them with a simple [LINK] tag
        text = re.sub(r'http\S+|www\S+', ' [LINK] ', text)
        # Remove symbols and numbers, leaving only regular letters
        text = re.sub(r'[^a-z\s]', ' ', text)
        return " ".join(text.split())

    def _train_engine(self):
        # We give the computer some examples of what scams and normal emails look like
        training_data = {
            "text": [
                "urgent bank account verification required now",
                "you won a million dollar prize click here",
                "account suspended login to secure your funds",
                "verify your password to avoid deactivation",
                "claim your free reward immediately",
                "your security has been breached click link",
                "meeting tomorrow at 10 am in the conference room",
                "please review the attached project report",
                "lunch plans for friday afternoon at the cafe",
                "happy birthday have a wonderful day",
                "your order has been delivered successfully",
                "can we reschedule our call to next week?"
            ],
            # 1 means Phishing (Scam), 0 means Safe
            "label": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        }
        df = pd.DataFrame(training_data)
        df['cleaned'] = df['text'].apply(self._clean_text)

        # Train the "brain" using the examples above
        X = self.vectorizer.fit_transform(df['cleaned'])
        self.model.fit(X, df['label'])

    def analyze_metadata(self, raw_text):
        # Check for sneaky things like too many links or using too many !! marks
        report = {
            "links": len(re.findall(r'http\S+', raw_text)),
            "urgency_marks": raw_text.count("!") + raw_text.count("?"),
            # Check if the email is "shouting" in ALL CAPS
            "caps_ratio": sum(1 for c in raw_text if c.isupper()) / len(raw_text) if len(raw_text) > 0 else 0
        }
        return report

    def get_risk_comment(self, is_phishing, prob, keywords, meta):
        # Pick a simple message to explain why the email is dangerous or safe
        if not is_phishing:
            if meta['links'] == 0:
                return "✅ This looks like a normal message. No red flags found."
            return "✅ Appears safe, but be careful with the links inside."

        # Explain which specific scam type was found
        if "bank" in keywords or "login" in keywords:
            return "🚩 DANGER: This is trying to steal your login or bank details."
        if "win" in keywords or "prize" in keywords:
            return "💰 SCAM: This is a fake prize offer used to trick you."
        if meta['urgency_marks'] > 5:
            return "⚠️ ALERT: This email is trying to make you panic with too many '!!!' marks."
        return "🛑 SUSPICIOUS: This looks like a common scam email pattern."

    def run_analysis(self, raw_email):
        # Step-by-step: Clean the text, check for red flags, then ask the AI's opinion
        cleaned = self._clean_text(raw_email)
        meta = self.analyze_metadata(raw_email)

        # Use the "brain" to get a risk percentage
        vec = self.vectorizer.transform([cleaned])
        prob = self.model.predict_proba(vec)[0][1] * 100

        # Check if any of our "bad words" are in the email
        found_flags = [w for w in self.risk_keywords if w in cleaned]

        # Decide if it's a scam based on the AI score and word count
        is_phishing = prob > 45 or len(found_flags) >= 2 or meta['links'] > 2

        # Get the simple explanation message
        comment = self.get_risk_comment(is_phishing, prob, found_flags, meta)

        return {
            "verdict": "🛑 SCAM DETECTED" if is_phishing else "✅ SAFE EMAIL",
            "confidence": round(prob, 2),
            "comment": comment,
            "flags": found_flags,
            "meta": meta
        }


# ============================================================
# 🖥️ START PROGRAM
# ============================================================

def main():
    detector = PhishingDetector()
    print("\n" + "=" * 60)
    print("        🛡️  EASY EMAIL SAFETY CHECKER")
    print("=" * 60)
    print("Paste your email below. Type 'PROCESS' on a new line to check it.")

    while True:
        lines = []
        print("\n📥 [PASTE EMAIL HERE...]")
        while True:
            line = input()
            if line.upper() == "PROCESS": break
            if line.upper() == "EXIT": return
            lines.append(line)

        full_text = "\n".join(lines)
        if not full_text.strip(): continue

        print("\nChecking the email... please wait.")
        time.sleep(1)

        report = detector.run_analysis(full_text)

        # Show the results simply
        print("\n" + "—" * 60)
        print(f"RESULT      : {report['verdict']}")
        print(f"RISK LEVEL  : {report['confidence']}%")

        # Show a simple bar to represent risk

        meter = int(report['confidence'] / 10)
        print(f"RISK METER  : [{'#' * meter}{'-' * (10 - meter)}]")

        print(f"WHY?        : {report['comment']}")

        if report['flags']:
            print(f"BAD WORDS   : {', '.join(report['flags'])}")

        print(f"LINKS FOUND : {report['meta']['links']}")
        print("—" * 60)


if __name__ == "__main__":
    main()