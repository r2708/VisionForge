# VisionForge Dataset – Issues List

## Critical Issues

1. **Hugging Face Dataset Viewer Error:** Dataset schema mismatch is causing `DatasetGenerationError` and `CastError`.
2. **Inconsistent JSONL Schema:** Record column names and declared dataset features do not match.
3. **Null-Type Inference:** Several nested fields are inferred as `list<null>` instead of their expected data types.
4. **Missing Schema Fields:** Actual records contain `image` and `text` columns that are missing from the declared schema.
5. **Failed Alignment Records Accepted:** Records with failed image-text alignment verification are still marked as accepted.
6. **Unverified Image Licensing:** Original image licenses are not individually verified, creating copyright and redistribution risks.

## High-Priority Issues

7. **Generic Captions:** Captions are generated using templates instead of detailed image understanding.
8. **Insufficient Caption Accuracy:** Some captions do not accurately describe the actual image content.
9. **Non-Independent Caption Verification:** CLIP-based signals are used for both annotation and verification.
10. **Watermarked Images Accepted:** Images flagged as potentially watermarked are still included in accepted records.
11. **Unreliable Quality Scoring:** High overall quality scores can coexist with very low aesthetic or alignment scores.
12. **Cross-Shard Near-Duplicate Detection:** Near-duplicate images are not checked globally across all shards.
13. **Potential Train-Test Leakage:** Similar or duplicate images may appear across training, validation, and test splits.

## Medium-Priority Issues

14. **Incomplete OCR Annotations:** Text present inside images may not be sufficiently extracted or represented.
15. **Limited Object Annotations:** Object and relationship annotations contain incomplete or null values.
16. **Insufficient Human Verification:** No clearly established human-reviewed benchmark covering the dataset is documented.
17. **Limited Category Diversity Reporting:** More detailed category distribution and imbalance reporting is needed.
18. **Incomplete Metadata Validation:** Optional fields and nested metadata need consistent type and value validation.

---

**Total: 18 identified issues.**
