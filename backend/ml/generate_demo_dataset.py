"""Generate a larger 3-class synthetic dataset for pipeline demonstrations only.

This is not a factual-news corpus and must never be used to claim real-world
classifier accuracy. Use a licensed, externally labeled dataset for that.
"""
import csv
from itertools import product
from pathlib import Path

OUTPUT = Path(__file__).resolve().parents[1] / "data" / "demo_news_large.csv"

REAL_SUBJECTS = [
    "the city council", "the public library", "the transit agency", "the county health department",
    "the school board", "the parks department", "the water utility", "the university",
    "the airport authority", "the fire service", "the museum", "the community centre",
]
REAL_ACTIONS = [
    "published meeting minutes", "released a service timetable", "announced a public consultation",
    "issued seasonal safety guidance", "posted a maintenance schedule", "updated registration instructions",
    "released an annual report", "announced a community programme", "published a budget summary",
    "opened applications for a grant programme", "published accessibility information", "reported scheduled repairs",
]
REAL_DETAILS = [
    "on its official website with dates and contact information",
    "after a recorded public meeting and formal vote",
    "with a link to supporting documents and the full agenda",
    "in a notice that identifies the responsible department",
    "alongside a downloadable schedule and service map",
    "with clear eligibility rules and an application deadline",
    "and invited residents to provide feedback through its public portal",
    "with a named spokesperson and an official press release",
    "including practical advice and links to verified resources",
    "with a published implementation date and a help phone number",
]

FAKE_SUBJECTS = [
    "an anonymous message", "a viral social-media post", "an unnamed insider", "a forwarded chain message",
    "a sensational blog", "an unverified video caption", "a secret account", "a mysterious website",
    "a copied online rumour", "a dramatic headline", "a cryptic post", "a message without a source",
]
FAKE_CLAIMS = [
    "claims every hospital will replace doctors with robots tonight",
    "says the moon broadcasts signals that control human thoughts",
    "promises a fruit can replace every medicine within one day",
    "alleges household appliances secretly alter election results",
    "says tap water turns into gold at midnight",
    "claims vegetables transmit hidden radio broadcasts",
    "announces that schools will replace teachers with talking plants",
    "says a phone update can read and sell people’s dreams",
    "claims the city council voted to ban clouds",
    "promises a bank will give unlimited money to first callers",
    "says an ancient statue predicts lottery numbers",
    "claims a new law requires everyone to wear purple on Fridays",
]
FAKE_DETAILS = [
    "but gives no named source, official statement, or evidence",
    "and urges people to share it before authorities remove the post",
    "while claiming that all journalists are hiding the truth",
    "and links only to an unverified form",
    "without identifying an agency, date, location, or credible report",
    "while saying ordinary fact-checking cannot be trusted",
    "and provides no documents that can be independently checked",
    "but relies entirely on an unnamed insider",
    "and says the evidence was deliberately deleted",
    "without citing any accountable institution",
]

VERIFY_SUBJECTS = [
    "a social-media post", "a local blog", "a forwarded message", "an online discussion",
    "a community group", "a screenshot shared online", "a video caption", "an unverified account",
]
VERIFY_CLAIMS = [
    "says the city may introduce free public transport next year",
    "claims a new health programme could start soon",
    "reports that a school policy might change next term",
    "suggests a local business may close next month",
    "says a proposed law could affect household bills",
    "claims a public event may be cancelled because of weather",
    "reports that officials are considering a new road plan",
    "suggests a university may add a new scholarship programme",
]
VERIFY_DETAILS = [
    "but no official decision has been published yet",
    "and includes a screenshot that cannot be independently verified",
    "while the available information is incomplete and may change",
    "but does not link to a primary source or final announcement",
    "and says readers should wait for a confirmed public notice",
    "although the responsible organization has not confirmed the claim",
    "but gives no date, named official, or verifiable document",
    "and should be checked against an authoritative source before sharing",
]

def main() -> None:
    rows: list[dict[str, str]] = []
    for index, (subject, action, detail) in enumerate(product(REAL_SUBJECTS, REAL_ACTIONS, REAL_DETAILS)):
        if index >= 300:
            break
        rows.append({"title": f"Official update: {action.title()}", "text": f"{subject.title()} {action} {detail}.", "label": "REAL"})
    for index, (subject, claim, detail) in enumerate(product(VERIFY_SUBJECTS, VERIFY_CLAIMS, VERIFY_DETAILS)):
        if index >= 300:
            break
        rows.append({"title": "Unconfirmed local claim", "text": f"{subject.capitalize()} {claim} {detail}.", "label": "UNVERIFIED"})
    for index, (subject, claim, detail) in enumerate(product(FAKE_SUBJECTS, FAKE_CLAIMS, FAKE_DETAILS)):
        if index >= 300:
            break
        rows.append({"title": "Unverified viral claim", "text": f"{subject.capitalize()} {claim} {detail}.", "label": "FAKE"})
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["title", "text", "label"])
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} synthetic demonstration rows to {OUTPUT}")

if __name__ == "__main__":
    main()
