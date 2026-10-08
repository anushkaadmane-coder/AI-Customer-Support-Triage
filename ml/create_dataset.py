import pandas as pd
import random
import os

random.seed(42)

# ============================================================
# CUSTOMER SUPPORT TICKET DATA
# ============================================================

categories = {
    "Billing": [
        "I have a question about my subscription payment",
        "I was charged for my subscription",
        "I need help with my invoice",
        "I want to update my payment information",
        "I need information about my billing",
        "There is an issue with my monthly payment",
        "I have a question about a transaction",
        "I need help understanding my bill",
        "My subscription payment needs to be checked",
        "I need assistance with my recent invoice",
        "I want to understand a charge on my account",
        "I need help with my payment method",
        "I have a problem with my subscription charge",
        "My invoice contains an incorrect amount",
        "I need assistance with a recent transaction",
        "I have a question about my refund",
        "I need help changing my billing details",
        "I want to review my recent payment",
        "There is a problem with my billing information",
        "I need support with my subscription charges"
    ],

    "Technical": [
        "The application is not working correctly",
        "The website is showing an error",
        "The application keeps crashing",
        "I cannot upload a file",
        "The page is not loading",
        "The application is running slowly",
        "I am receiving an error message",
        "The website stopped responding",
        "The application has a technical problem",
        "I cannot access a particular feature",
        "The system is behaving unexpectedly",
        "The website keeps freezing",
        "The application stopped working",
        "I am having trouble loading the dashboard",
        "A technical error is preventing me from continuing",
        "The upload function is not working",
        "The website is taking too long to respond",
        "I am experiencing a system error",
        "The application is displaying an unexpected error",
        "The service is not functioning properly"
    ],

    "Account": [
        "I forgot my password",
        "I cannot log into my account",
        "I need to change my email address",
        "My account has been locked",
        "I want to update my profile",
        "I cannot access my account",
        "I need help resetting my password",
        "I want to change my account details",
        "My login is not working",
        "I need help with account access",
        "I want to update my personal information",
        "I cannot reset my password",
        "My account access has been restricted",
        "I need assistance with my login",
        "I want to modify my profile information",
        "I am having trouble signing into my account",
        "I need help recovering my account",
        "I want to delete my account",
        "My account credentials are not working",
        "I need assistance managing my account"
    ],

    "Product": [
        "How does this product feature work",
        "I want information about the premium plan",
        "How can I upgrade my subscription",
        "I need help understanding a product feature",
        "Which subscription plan should I choose",
        "Can I use this feature with my current plan",
        "I want to learn more about the product",
        "What features are included in my plan",
        "How can I use this feature",
        "I need information about the available plans",
        "Is this feature available on my subscription",
        "Can you explain this product feature",
        "I want to compare the available plans",
        "How does the premium version work",
        "I need help choosing a plan",
        "Can I upgrade my current plan",
        "I want more information about the product",
        "What is included in the basic plan",
        "Can you explain how this feature works",
        "I have a question about product functionality"
    ]
}


# ============================================================
# URGENCY SIGNALS
# ============================================================

urgency_phrases = {

    "Low": [
        "when you have time",
        "there is no rush",
        "I would like some information",
        "this is not urgent",
        "whenever possible",
        "I can wait",
        "at your convenience"
    ],

    "Medium": [
        "I would appreciate help soon",
        "this has been happening for a while",
        "I need help with this issue",
        "please look into this",
        "I would like this resolved soon",
        "this is becoming inconvenient",
        "please assist me"
    ],

    "High": [
        "I need this resolved as soon as possible",
        "this is affecting my work",
        "I cannot continue using the service",
        "I need immediate assistance",
        "this is causing a major problem",
        "please prioritize this issue",
        "I urgently need help"
    ],

    "Critical": [
        "this is an emergency",
        "my account may be compromised",
        "there are unauthorized transactions",
        "I believe someone has accessed my account",
        "the entire service is unavailable",
        "this is a critical issue",
        "I need immediate emergency assistance"
    ]
}


# ============================================================
# RESOLUTION INFORMATION
# ============================================================

resolutions = {
    "Billing": "Billing support",
    "Technical": "Technical support",
    "Account": "Account support",
    "Product": "Product support"
}


# ============================================================
# GENERATE DATASET
# ============================================================

data = []
ticket_id = 1

for category, base_tickets in categories.items():

    for base_ticket in base_tickets:

        for urgency, phrases in urgency_phrases.items():

            # Pick an urgency-specific phrase
            phrase = random.choice(phrases)

            # Create ticket
            ticket_text = f"{base_ticket}. {phrase}."

            data.append([
                f"TKT-{ticket_id:04d}",
                ticket_text,
                category,
                urgency,
                resolutions[category]
            ])

            ticket_id += 1


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(
    data,
    columns=[
        "ticket_id",
        "ticket_text",
        "category",
        "urgency",
        "resolution"
    ]
)


# Shuffle
df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# SAVE DATASET
# ============================================================

os.makedirs("../dataset", exist_ok=True)

output_path = "../dataset/tickets.csv"

df.to_csv(
    output_path,
    index=False
)


# ============================================================
# DISPLAY INFORMATION
# ============================================================

print("=" * 60)
print("DATASET CREATED SUCCESSFULLY")
print("=" * 60)

print(f"\nTotal tickets: {len(df)}")

print("\nCategories:")
print(df["category"].value_counts())

print("\nUrgency:")
print(df["urgency"].value_counts())

print("\nSample tickets:")

print(
    df[
        ["ticket_text", "category", "urgency"]
    ].head(10).to_string(index=False)
)

print(f"\nSaved to: {output_path}")