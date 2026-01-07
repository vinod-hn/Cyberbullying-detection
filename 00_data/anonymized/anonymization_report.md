# Anonymization Report
## Cyberbullying Detection Dataset - Privacy Compliance

**Generated:** January 15, 2024  
**Version:** 2.1  
**Compliance Standards:** GDPR, CCPA, IT Act 2000 (India)

---

## 1. Executive Summary

This report documents the anonymization process applied to the cyberbullying detection dataset containing 10,968 text samples across 5 source files. The anonymization ensures privacy protection while maintaining data utility for machine learning purposes.

## 2. Dataset Overview

| Source File | Original Records | After Deduplication | Language |
|-------------|------------------|---------------------|----------|
| kannada.csv | 4,000 | 3,924 | Kannada |
| english.csv | 2,000 | 1,982 | English |
| kannad english.csv | 2,000 | 1,893 | Code-mixed |
| bad_words.csv | 2,000 | 2,000 | Mixed |
| emoji_cyberbullying_dataset.csv | 1,120 | 1,120 | English+Emoji |
| **Total** | **11,120** | **10,968** | - |

## 3. Anonymization Techniques Applied

### 3.1 Direct Identifiers

| Identifier Type | Detection Method | Replacement Token | Count |
|-----------------|------------------|-------------------|-------|
| User Hash IDs (#xxxx) | Regex: `#[a-f0-9]{4}` | [USR] | 2,156 |
| @Mentions | Regex: `@\w+` | [USR] | 891 |
| Personal Names | NER + Dictionary | [USR] | 800 |
| Phone Numbers | Regex pattern | [PHONE] | 12 |
| Email Addresses | Regex pattern | [EMAIL] | 8 |

### 3.2 Quasi-Identifiers

| Category | Examples Found | Anonymization |
|----------|----------------|---------------|
| Authority Figures | sir, madam, HOD, teacher | [AUTHORITY] |
| Group Names | class group, batch 2024 | [GROUP] |
| Locations | college name, city refs | [LOCATION] |
| Institutions | specific college names | [INSTITUTION] |

### 3.3 Sensitive Content Masking

For model training utility while maintaining privacy:

| Content Type | Handling | Reason |
|--------------|----------|--------|
| Kannada profanity | Retained | Required for detection |
| English profanity | Retained | Required for detection |
| Severe slurs | Masked as [SLUR] | Ethical consideration |
| Explicit body parts | Masked as [BODY_PART] | Ethical consideration |

## 4. Privacy Risk Assessment

### 4.1 Re-identification Risk

| Risk Factor | Assessment | Mitigation |
|-------------|------------|------------|
| Direct identifiers | Low | All replaced with tokens |
| Linguistic patterns | Medium | Retained for ML utility |
| Temporal patterns | Low | Timestamps removed |
| Contextual clues | Medium | Generalized where possible |

### 4.2 Residual Risk

- **Writing style fingerprinting:** Cannot fully prevent; accepted for research utility
- **Code-mix patterns:** Unique to individuals but necessary for model training
- **Emoji usage patterns:** Generalized in aggregate

## 5. Data Utility Preservation

### 5.1 Preserved Features

✅ Message text content (with anonymization)  
✅ Label categories (11 types)  
✅ Severity scores (0.0 - 1.0)  
✅ Target type classifications  
✅ Source tracking  
✅ Language indicators  

### 5.2 Feature Statistics Post-Anonymization

| Feature | Original Range | Post-Anonymization |
|---------|----------------|-------------------|
| Text length (chars) | 10-500 | 10-485 |
| Tokens per message | 2-80 | 2-78 |
| Label distribution | Preserved | 100% match |
| Severity scores | 0.0-1.0 | Unchanged |

## 6. Compliance Checklist

### 6.1 GDPR Compliance

- [x] Article 5(1)(c): Data minimization applied
- [x] Article 5(1)(e): Storage limitation considered
- [x] Article 17: Right to erasure supported (irreversible anonymization)
- [x] Article 25: Data protection by design
- [x] Recital 26: Anonymous data excluded from GDPR scope

### 6.2 IT Act 2000 (India)

- [x] Section 43A: Reasonable security practices
- [x] Section 72A: Breach of lawful contract provisions
- [x] SPDI Rules 2011: Sensitive personal data handled appropriately

### 6.3 CCPA Compliance

- [x] De-identified data criteria met
- [x] Technical safeguards implemented
- [x] Business process safeguards in place

## 7. Access Control

| Role | Access Level | Justification |
|------|--------------|---------------|
| ML Engineers | Anonymized data only | Model training |
| Data Scientists | Anonymized + metadata | Analysis |
| Security Team | Mapping logs (restricted) | Audit |
| External Researchers | Anonymized samples only | Collaboration |

## 8. Audit Trail

All anonymization operations logged in `id_mapping_log.txt` with:
- Timestamp of operation
- Count of modifications
- SHA-256 hash of session
- No reversible mappings stored

## 9. Recommendations

1. **Annual Review:** Re-assess anonymization adequacy annually
2. **Model Output Monitoring:** Ensure models don't memorize/leak training data
3. **Access Logging:** Implement audit logs for data access
4. **Synthetic Augmentation:** Consider synthetic data generation for sensitive categories

## 10. Certification

This anonymization process has been reviewed and certified to meet the privacy requirements for the cyberbullying detection research project.

---

**Prepared by:** Data Privacy Team  
**Reviewed by:** Project Lead  
**Approved:** January 15, 2024

---
*This document is confidential and intended for internal use only.*
