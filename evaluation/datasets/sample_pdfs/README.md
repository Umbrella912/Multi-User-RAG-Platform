# Sample PDFs for evaluation

Drop sample documents (PDF, TXT, MD, DOCX) into this directory, then build the
evaluation dataset and run the evaluators:

```bash
# 1. Add your sample files here (this directory).

# 2. Build 25 Q/A/ground-truth triples from them
python evaluation/build_dataset.py

# 3. Run RAGAS against the pipeline
python evaluation/run_ragas.py --pdf-dir evaluation/datasets/sample_pdfs

# 4. Compare retrieval configurations
python evaluation/compare_configs.py --pdf-dir evaluation/datasets/sample_pdfs
```

## Tips for good evaluation datasets

- **Use 3–5 documents** with clearly answerable facts (policies, specs,
  FAQs, product descriptions) — questions that have a definite answer in the
  text.
- **Aim for 25 questions** (the default count); the builder spreads them
  across documents automatically.
- **Ground truths matter**: RAGAS context_recall and the config comparator
  score retrieval against the `contexts`/`ground_truths` you provide. For
  maximum signal, each question's ground truth should be a verbatim sentence
  from the source document.
- You can also hand-author triples using `template_qa.json` (in this
  directory) instead of running the builder.
