# Data Quality Report
## Cyberbullying Detection Dataset

**Generated:** January 15, 2024  
**Dataset Version:** 1.0  
**Total Samples:** 10,968 (after deduplication)

---

## 1. Executive Summary

This report assesses the quality of the cyberbullying detection dataset across multiple dimensions including completeness, consistency, accuracy, and fitness for ML model training.

| Quality Dimension | Score | Status |
|-------------------|-------|--------|
| Completeness | 98.7% | ✅ Good |
| Consistency | 94.2% | ✅ Good |
| Accuracy | 92.5% | ✅ Good |
| Uniqueness | 98.6% | ✅ Good |
| Timeliness | N/A | ✅ Current |
| **Overall Quality** | **96.0%** | ✅ **Production Ready** |

---

## 2. Data Completeness

### 2.1 Missing Values Analysis

| Field | Total Records | Missing | Percentage | Status |
|-------|---------------|---------|------------|--------|
| message | 10,968 | 0 | 0.00% | ✅ |
| label | 10,968 | 0 | 0.00% | ✅ |
| target_type | 10,968 | 142 | 1.29% | ✅ |
| severity | 10,968 | 0 | 0.00% | ✅ |
| severity_score | 10,968 | 0 | 0.00% | ✅ |
| source | 10,968 | 0 | 0.00% | ✅ |

### 2.2 Field Population Rate

- **Required fields:** 100% complete
- **Optional fields:** 98.7% complete
- **Overall completeness:** 98.7%

---

## 3. Data Consistency

### 3.1 Label Consistency

All labels normalized to lowercase:
- Original variations: "Insult", "insult", "INSULT" → "insult"
- Standardized to 11 unique labels
- No undefined/unknown labels

### 3.2 Severity Score Consistency

| Check | Result | Status |
|-------|--------|--------|
| Range (0.0 - 1.0) | All within range | ✅ |
| Numeric type | All float | ✅ |
| Precision | 2 decimal places | ✅ |
| Score-Label alignment | 94.2% aligned | ✅ |

### 3.3 Score-Label Alignment Analysis

Expected severity ranges by label:

| Label | Expected Range | Actual Mean | Alignment |
|-------|----------------|-------------|-----------|
| neutral | 0.00 - 0.15 | 0.04 | ✅ |
| insult | 0.20 - 0.60 | 0.38 | ✅ |
| harassment | 0.40 - 0.70 | 0.52 | ✅ |
| threat | 0.60 - 0.95 | 0.78 | ✅ |
| exclusion | 0.30 - 0.60 | 0.42 | ✅ |
| aggression | 0.40 - 0.70 | 0.55 | ✅ |
| toxicity | 0.35 - 0.65 | 0.48 | ✅ |
| stalking | 0.50 - 0.80 | 0.62 | ✅ |
| sexual_harassment | 0.60 - 0.90 | 0.71 | ✅ |
| hate | 0.60 - 0.90 | 0.76 | ✅ |
| cyberstalking | 0.60 - 0.90 | 0.79 | ✅ |

---

## 4. Data Accuracy

### 4.1 Sample Quality Review

- **Samples reviewed:** 1,097 (10% random sample)
- **Correct labels:** 1,015 (92.5%)
- **Ambiguous but acceptable:** 58 (5.3%)
- **Incorrect labels:** 24 (2.2%)

### 4.2 Common Accuracy Issues

1. **Borderline harassment/aggression** (12 cases)
   - Action: Re-reviewed with senior annotator
   
2. **Sarcasm misinterpretation** (7 cases)
   - Action: Flagged for model training awareness
   
3. **Cultural context missing** (5 cases)
   - Action: Added Kannada-specific guidance

### 4.3 Text Quality

| Check | Pass Rate | Status |
|-------|-----------|--------|
| Encoding (UTF-8) | 100% | ✅ |
| Minimum length (5 chars) | 99.8% | ✅ |
| Maximum length (500 chars) | 100% | ✅ |
| No HTML/special formatting | 99.9% | ✅ |
| Readable text | 99.5% | ✅ |

---

## 5. Data Uniqueness

### 5.1 Duplicate Detection

| Stage | Records | Duplicates | Unique |
|-------|---------|------------|--------|
| Raw combined | 11,120 | 152 | 10,968 |
| After dedup | 10,968 | 0 | 10,968 |

### 5.2 Near-Duplicate Analysis

- **Exact duplicates removed:** 152
- **Near-duplicates (>90% similarity):** 87 pairs
  - Reviewed and kept: Different contexts/labels
- **Template-based duplicates:** 0

### 5.3 Source Overlap

| Source Pair | Overlap | Notes |
|-------------|---------|-------|
| english ↔ kannada | 0 | No overlap |
| english ↔ bad_words | 23 | Similar insults, different labels |
| kannada ↔ codemix | 45 | Transliteration variants |
| emoji ↔ others | 12 | Same text, different emoji |

---

## 6. Class Distribution Quality

### 6.1 Imbalance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Imbalance Ratio | 10.9:1 | Moderate imbalance |
| Gini Index | 0.342 | Some imbalance |
| Shannon Entropy | 3.12 | Good diversity |

### 6.2 Recommendations for Model Training

1. **Oversampling:** SMOTE for minority classes (cyberstalking, hate)
2. **Class weights:** Inversely proportional to frequency
3. **Stratified splitting:** Implemented ✅
4. **Threshold tuning:** Required for deployment

---

## 7. Language Quality

### 7.1 Language Distribution

| Language | Count | Percentage |
|----------|-------|------------|
| Kannada | 4,000 | 36.5% |
| English | 3,120 | 28.4% |
| Code-mixed | 2,728 | 24.9% |
| Mixed (unclear) | 1,120 | 10.2% |

### 7.2 Script Consistency

| Script | Expected | Found | Status |
|--------|----------|-------|--------|
| Latin (English) | Yes | Yes | ✅ |
| Kannada Unicode | Yes | Minimal | ⚠️ |
| Romanized Kannada | Yes | Majority | ✅ |
| Emoji | Yes | Yes | ✅ |

**Note:** Most Kannada content is in romanized form (transliterated), not native Kannada script.

---

## 8. Fitness for Purpose

### 8.1 ML Training Readiness

| Requirement | Status | Notes |
|-------------|--------|-------|
| Sufficient volume | ✅ | 10,968 samples |
| Label quality | ✅ | IAA > 0.75 |
| Text diversity | ✅ | Multiple sources |
| Real-world representative | ✅ | Social media context |
| Privacy compliant | ✅ | Anonymized |

### 8.2 Known Limitations

1. **Regional bias:** Karnataka-centric Kannada
2. **Platform bias:** Academic/student group context
3. **Temporal bias:** Current language patterns (may shift)
4. **Demographic gaps:** Adult workplace harassment underrepresented

### 8.3 Recommended Use Cases

✅ **Appropriate:**
- Student group moderation
- Educational platform safety
- Regional language bullying detection
- Code-mixed NLP research

⚠️ **Use with caution:**
- General social media (needs fine-tuning)
- Non-Kannada regional languages
- Professional/workplace contexts

❌ **Not recommended:**
- Child-specific content (different patterns)
- Formal communication channels
- Non-Indian English contexts

---

## 9. Data Lineage

### 9.1 Source Tracking

All samples include `source` field tracking origin:
- english.csv → source="english"
- kannada.csv → source="kannada"
- kannad english.csv → source="codemix"
- bad_words.csv → source="bad_words"
- emoji_cyberbullying_dataset.csv → source="emoji"

### 9.2 Processing History

1. Raw data collection → 11,120 samples
2. Deduplication → 10,968 samples
3. Label normalization → Standardized
4. Severity validation → Verified
5. Stratified splitting → 70/15/15

---

## 10. Quality Certification

| Checkpoint | Verified By | Date | Status |
|------------|-------------|------|--------|
| Completeness | Data Team | 2024-01-10 | ✅ |
| Consistency | QA Team | 2024-01-12 | ✅ |
| Accuracy (IAA) | Annotators | 2024-01-13 | ✅ |
| Privacy | Privacy Team | 2024-01-14 | ✅ |
| Final Review | Project Lead | 2024-01-15 | ✅ |

---

## 11. Action Items

### Completed
- [x] Remove exact duplicates
- [x] Normalize labels
- [x] Validate severity scores
- [x] Anonymize identifiers
- [x] Create stratified splits

### Pending
- [ ] Add more cyberstalking samples (underrepresented)
- [ ] Include native Kannada script samples
- [ ] Expand to workplace context
- [ ] Periodic re-annotation for quality

---

**Report Prepared By:** Data Quality Team  
**Approved By:** Project Lead  
**Next Review Date:** July 2024

---
*This report is for internal use in the Cyberbullying Detection project.*
