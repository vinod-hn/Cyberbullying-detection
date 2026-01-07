# Annotation Guidelines
## Cyberbullying Detection Dataset - Kannada/English

**Version:** 2.0  
**Last Updated:** January 15, 2024  
**Languages:** English, Kannada, Kannada-English Code-Mixed

---

## 1. Overview

This document provides guidelines for annotating text messages for cyberbullying detection. Annotators must classify each message into one of 11 categories and assign severity scores.

## 2. Label Categories

### 2.1 Neutral
**Definition:** Messages with no harmful intent, normal conversations.

**Examples:**
- "Can someone share the notes from today?"
- "What time is the assignment due?"
- "naale 9 gante lab ge bartira" (Will you come to lab at 9 tomorrow?)

**Criteria:**
- No insults, threats, or negative targeting
- Routine communication
- Friendly or professional tone

---

### 2.2 Insult
**Definition:** Messages that demean, belittle, or attack someone's character, intelligence, or worth.

**Examples:**
- "stop talking rubbish"
- "you make everything worse"
- "nee sakkat dumb idiya" (You are extremely dumb)
- "nee full time late bartiya" (You always come late)

**Criteria:**
- Direct or indirect attack on personal qualities
- Name-calling or derogatory language
- Mocking or ridiculing

**Severity Indicators:**
- Low: Mild insults (stupid, dumb, lazy)
- Medium: Personal attacks (worthless, pathetic)
- High: Severe profanity-based insults

---

### 2.3 Harassment
**Definition:** Repeated unwanted contact, persistent annoyance, or boundary violations.

**Examples:**
- "dont keep asking the same thing"
- "why are you so behind me"
- "dont message me late night"
- "how many times should I say no"

**Criteria:**
- Unwanted persistent contact
- Violation of stated boundaries
- Causing distress through repetition

---

### 2.4 Threat
**Definition:** Messages expressing intent to cause harm, damage, or negative consequences.

**Examples:**
- "you wont get away with this"
- "dont think this will go unnoticed"
- "I will report this to sir"
- "consequences face maadu" (Face the consequences)

**Criteria:**
- Explicit or implied threat of harm
- Intimidation
- Warnings of negative action

**Severity Indicators:**
- Medium: Social consequences (reporting, telling others)
- High: Physical harm implications
- Critical: Direct violence/death threats

---

### 2.5 Exclusion
**Definition:** Deliberately leaving someone out, isolating, or making them feel unwelcome.

**Examples:**
- "you are not required"
- "this is between us only"
- "this is only for selected people"
- "ivanna group inda remove maadona" (Let's remove him from the group)

**Criteria:**
- Explicit exclusion from groups/activities
- Creating in-group/out-group dynamics
- Making someone feel unwanted

---

### 2.6 Aggression
**Definition:** Hostile, confrontational behavior that may not be a direct threat but creates a hostile environment.

**Examples:**
- "Stop it, keep quiet please"
- "go away from here for real"
- "go mind your work"
- "dont interfere in this"

**Criteria:**
- Hostile tone
- Confrontational language
- Dismissive aggression

---

### 2.7 Toxicity
**Definition:** Generally negative, harmful atmosphere-creating messages without specific targeting.

**Examples:**
- "this environment is really bad"
- "this place feels unhealthy"
- "this place is full of nonsense"

**Criteria:**
- Negative generalizations
- Complaint-based negativity
- Creating pessimistic atmosphere

---

### 2.8 Stalking
**Definition:** Tracking, monitoring, or excessive attention to someone's activities.

**Examples:**
- "why do you remember my timings"
- "why are you noticing my movements"
- "why do you keep observing me"

**Criteria:**
- Unwanted monitoring behavior
- Privacy violations
- Tracking online/offline activities

---

### 2.9 Cyberstalking
**Definition:** Digital-specific stalking behaviors involving online tracking and monitoring.

**Examples:**
- "nanna status na yavaglu check madthiya" (You always check my status)
- "prati dina reason illade ping madbedi" (Don't ping without reason every day)
- "Online status always check madthidya"

**Criteria:**
- Online activity monitoring
- Social media stalking
- Digital harassment patterns

---

### 2.10 Hate
**Definition:** Prejudice-based attacks targeting groups or individuals based on identity characteristics.

**Examples:**
- "your kind spoils the environment"
- "people like you make everything dirty"
- "your type always ruins things"

**Criteria:**
- Group-based targeting
- Prejudice or discrimination
- Identity-based attacks (gender, ethnicity, religion)

---

### 2.11 Sexual Harassment
**Definition:** Unwanted sexual comments, advances, or references.

**Examples:**
- "sexy bro" (inappropriate context)
- Messages with sexual profanity
- Unwanted romantic/sexual advances

**Criteria:**
- Sexual content without consent
- Objectification
- Sexual profanity directed at person

---

## 3. Severity Scoring

### 3.1 Scale: 0.0 to 1.0

| Range | Level | Description |
|-------|-------|-------------|
| 0.00 - 0.10 | Minimal | Neutral, no harm |
| 0.11 - 0.33 | Low | Mild negativity |
| 0.34 - 0.66 | Medium | Clear cyberbullying |
| 0.67 - 0.89 | High | Severe cyberbullying |
| 0.90 - 1.00 | Critical | Immediate intervention needed |

### 3.2 Factors Increasing Severity
- Direct personal targeting
- Use of profanity
- Threat of physical harm
- Repeated patterns
- Vulnerable target (appearance, disability references)
- Sexual content
- Group-based hate

### 3.3 Factors Decreasing Severity
- Ambiguous intent
- Possible humor/sarcasm
- Generic complaints
- First-time occurrence

---

## 4. Target Type Classification

| Target Type | Description | Examples |
|-------------|-------------|----------|
| individual | Specific person targeted | "You are stupid" |
| group | Group of people | "All of you are fake" |
| appearance | Physical appearance | "You are ugly" |
| academic | Academic performance | "You always fail" |
| gender | Gender-based | "Girls can't do this" |
| unknown | Cannot determine | Generic complaints |

---

## 5. Special Considerations

### 5.1 Code-Mixed Messages
- Consider both languages for context
- Kannada profanity may be more severe culturally
- English insults may be normalized in some contexts

### 5.2 Emoji Interpretation
- Consider emoji sentiment
- 🤡 = mockery (increase severity)
- 💢 = anger (increase severity)
- 😂 = context-dependent (may reduce or increase)

### 5.3 Cultural Context
- "machaa", "maga", "re" are friendly particles
- Same words can be friendly OR aggressive
- Context is critical

### 5.4 Edge Cases
- Playful banter: Check for bidirectional insults
- Quotes: If quoting someone else, label the quoted content
- Sarcasm: Consider literal interpretation if unclear

---

## 6. Annotation Process

1. Read the full message
2. Identify primary harmful behavior (if any)
3. Assign ONE label (most prominent)
4. Determine target type
5. Assign severity score
6. Flag uncertain cases for review

---

## 7. Quality Assurance

- 10% of annotations undergo double review
- Inter-annotator agreement threshold: 0.75 (Cohen's Kappa)
- Disputed cases resolved by senior annotator
- Regular calibration sessions

---

**Contact:** For questions, contact the annotation lead.
