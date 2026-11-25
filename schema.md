# Input contract
`records.csv`: one row per documented observation, with unique `record_id`, `timestamp` (prediction time), `outcome_timestamp` (later observation time), `brand`, `category`, `region`, `brand_value` (observed normalized 0–100), `risk_label` (observed binary screening outcome). Year is derived from timestamp.

11 structured columns: rating, review_count, helpfulness, price, trade_growth, tariff_volatility, trademark_status, sanctions_similarity, origin_documentation, content_restriction, year_offset. Blank numeric fields are unknown. Review count and price are log1p-transformed. Legal status/similarity/restriction and tariff volatility must be on documented rule-compatible scales; year_offset must be computed without future information.

6 reliability columns: text_present, review_volume_proxy, rating_consistency, image_present, metadata_completeness, provenance_weight. Presence flags 0/1, other scores finite, preferably in [0,1] with documented definitions.

`embeddings.npz`: NumPy arrays `record_id` (unicode), `text` (N × D_t float), `image` (N × D_v float). Actual feature embeddings must be derived only from evidence available at prediction time. Missing modality rows may have NaNs; only those rows are masked. No random embeddings are supplied.

## Source provenance to establish before execution
- Amazon Reviews’23: https://amazon-reviews-2023.github.io/ — commercial reviews and product metadata. Does not by itself identify export destinations or observed brand valuation.
- USPTO: https://huggingface.co/datasets/illeatmyhat/uspto-trademarks-bronze — trademark case records. Document entity linkage and event-time status; absence is not proof of invalidity.
- UN Comtrade: https://comtradeplus.un.org/DataAvailability — trade flows; does not establish brand-level export or tariff volatility by itself.
- OFAC: https://ofac.treasury.gov/sanctions-list-service — sanctions evidence; use historical snapshots as of prediction time, document name matching and provenance; similarity is not a confirmed match.

Maintain source version, access date, raw record identifiers, observation time, mapping keys, match policy, label definition and coverage for every field. The four sources do not necessarily provide origin documents, content restrictions, tariffs or independent outcomes. Obtain documented additional evidence where necessary, or retain missing values for features; never invent labels. The manuscript does not specify a validated cross-source join, so no automatic join is presented as the original method.
