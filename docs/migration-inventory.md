# Migration inventory

A file-by-file account of `DiogoRibeiro7/DiogoRibeiro7.github.io` as it stood at
the start of the migration, recording what each file is, where it belongs, and
how far it has moved.

The survey covers every tracked Python file, every notebook, the benchmark data
files under `assets/viz/`, and the references to those files from `_posts/`,
`_drafts/`, `_pages/`, `README.md`, and the site workflows. Generated caches
(`__pycache__`, `.ipynb_checkpoints`, `.jekyll-cache`, `_site`) are excluded.

## Classification

| Class | Meaning | Destination |
| --- | --- | --- |
| **W** — website infrastructure | builds, validates, or maintains the site | stays in the website repository |
| **R** — reproducibility code | models, simulations, benchmarks, figure generation | moves here |
| **A** — rendered publication asset | the PNG/JPEG a published article links to | stays in the website repository |
| **D** — derived data | benchmark output and similar generated records | moves here with provenance |
| **O** — obsolete | scratch or superseded work | deleted, not migrated |

Nothing was left in a "needs inspection" state: every file below was opened and
classified.

## Migration status vocabulary

`pending` — not started. `migrated` — present and tested here. `complete` —
migrated here *and* removed from the website repository. `outstanding` — still
in the website repository.

As of the website cleanup, every `R` row is `complete` except
`generate_figures.py`.

---

## 1. `assets/viz/` — figure generation, benchmarks, shared style

| Original path | Class | Destination | Status |
| --- | --- | --- | --- |
| `data_lake_benchmarks.py` | R | `engineering/data_lake_benchmarks.py` | **complete** |
| `data_lake_benchmarks.json` | D | `data/engineering/data_lake_benchmarks.json` | **complete** |
| `database_benchmarks.py` | R | `engineering/database_benchmarks.py` | **complete** |
| `database_benchmarks.json` | D | `data/engineering/database_benchmarks.json` | **complete** |
| `fetch_headers.py` | W | — | stays |
| `generate_2026_evidence_articles.py` | R | split across `statistics/`, `health/`, `physics/`, `engineering/` | **complete** |
| `generate_aspartame_dose_figures.py` | R | `health/aspartame_dose*.py` | **complete** |
| `generate_confidence_set_figures.py` | R | `statistics/confidence_sets*.py` | **complete** |
| `generate_coverage_draft_figures.py` | R | `statistics/`, `health/`, `engineering/` (three draft models) | **complete** |
| `generate_data_lake_figures.py` | R | `engineering/data_lake_figure.py` | **complete** |
| `generate_database_figures.py` | R | `engineering/database_figure.py` | **complete** |
| `generate_figures.py` | R | split by domain across the package | **outstanding — see section 7** |
| `generate_headers.py` | W | — | stays |
| `generate_hormone_testing_figures.py` | R | `health/hormone_testing*.py` | **complete** |
| `generate_inflammation_marker_figures.py` | R | `health/inflammation_markers*.py` | **complete** |
| `generate_leaky_gut_figures.py` | R | `health/leaky_gut*.py` | **complete** |
| `generate_microbiome_testing_figures.py` | R | `health/microbiome_testing*.py` | **complete** |
| `generate_parasite_testing_figures.py` | R | `health/parasite_testing*.py` | **complete** |
| `generate_poll_selection_figures.py` | R | `statistics/poll_selection*.py` | **complete** |
| `generate_pvalue_evidence_figures.py` | R | `statistics/pvalue_evidence*.py` | **complete** |
| `generate_quantum_observer_figures.py` | R | `physics/quantum_observer*.py` | **complete** |
| `generate_results_rhetoric_figures.py` | R | `health/results_rhetoric*.py` | **complete** |
| `generate_science_communication_figures.py` | R | `physics/`, `statistics/`, `health/` | **complete** |
| `generate_screening_survival_figures.py` | R | `health/screening_survival*.py` | **complete** |
| `generate_sequential_changepoint_figure.py` | R | `time_series/sequential_cusum*.py` | **complete** |
| `generate_testimonial_figures.py` | R | `statistics/testimonial_selection*.py` | **complete** |
| `house.mplstyle` | W + R | `common/house.mplstyle` | **migrated**; the website keeps its own copy |
| `housestyle.py` | W + R | `common/plotting.py` | **migrated**; the website keeps a reduced copy |

### Why the house style stays in both repositories

`housestyle.py` and `house.mplstyle` are a design specification, not scientific
source. `generate_headers.py` — which stays in the website repository, because
header images are site decoration rather than evidence — imports the palette
from it. The scientific consumers now use `blog_reproducibility.common.plotting`,
and the website copy is reduced to what header generation actually needs. No
scientific code is duplicated.

## 2. `tests/` — website tests and article tests

| Original path | Class | Destination | Status |
| --- | --- | --- | --- |
| `test_aspartame_dose_models.py` | R | `tests/health/test_aspartame_dose.py` | **complete** |
| `test_coverage_draft_models.py` | R | `tests/statistics/`, `tests/health/`, `tests/engineering/` | **complete** |
| `test_data_lake_benchmarks.py` | R | `tests/engineering/test_data_lake_benchmarks.py` | **complete** |
| `test_database_benchmarks.py` | R | `tests/engineering/test_database_benchmarks.py` | **complete** |
| `test_hormone_testing_models.py` | R | `tests/health/test_hormone_testing.py` | **complete** |
| `test_inflammation_marker_models.py` | R | `tests/health/test_inflammation_markers.py` | **complete** |
| `test_leaky_gut_models.py` | R | `tests/health/test_leaky_gut.py` | **complete** |
| `test_microbiome_testing_models.py` | R | `tests/health/test_microbiome_testing.py` | **complete** |
| `test_parasite_testing_models.py` | R | `tests/health/test_parasite_testing.py` | **complete** |
| `test_poll_selection_models.py` | R | `tests/statistics/test_poll_selection.py` | **complete** |
| `test_post_layout.py` | W | — | stays |
| `test_post_math_delimiters.py` | W | — | stays |
| `test_pvalue_evidence_models.py` | R | `tests/statistics/test_pvalue_evidence.py` | **complete** |
| `test_quantum_observer_models.py` | R | `tests/physics/test_quantum_observer.py` | **complete** |
| `test_results_rhetoric_models.py` | R | `tests/health/test_results_rhetoric.py` | **complete** |
| `test_science_communication_figures.py` | R | `tests/physics/`, `tests/statistics/`, `tests/health/` | **complete** |
| `test_screening_survival_models.py` | R | `tests/health/test_screening_survival.py` | **complete** |
| `test_sequential_cusum.py` | R | `tests/time_series/test_sequential_cusum.py` | **complete** |
| `test_sync_theme_assets.py` | W | — | stays |
| `test_testimonial_models.py` | R | `tests/statistics/test_testimonial_selection.py` | **complete** |

Once every `R` row is complete, the website still needs `requirements.txt` and
the `python-tests.yml` workflow: `test_post_layout.py`,
`test_post_math_delimiters.py`, and `test_sync_theme_assets.py` remain there.

## 3. `code/` — the downloadable examples directory

| Original path | Class | Destination | Status |
| --- | --- | --- | --- |
| `code/Untitled.ipynb` | O | — | delete |
| `code/.ipynb_checkpoints/Untitled-checkpoint.ipynb` | O | — | delete, never migrated |
| `code/michelson_morley.py` | O | — | delete |

Both files hold the same twenty lines: ten numbers labelled `# Hypothetical
data` in a box plot, with `plt.show()` and no output path. The notebook's second
cell is empty. No article references either file; the only link is the generic
"Example notebook" entry on `/code/`. They are scratch work under a meaningless
name, so they are deleted rather than migrated.

The roadmap's Michelson–Morley item is therefore not a migration. It is
implemented here from the physics instead: the fringe shift a stationary aether
predicts for a rotating interferometer, against the bound the 1887 apparatus
could resolve. That is recorded as new work, not as migrated code.

## 4. `scripts/` — site maintenance

| Original path | Class | Destination | Status |
| --- | --- | --- | --- |
| `check_internal_links.py` | W | — | stays |
| `sync_theme_assets.py` | W | — | stays |
| `validate_front_matter.rb` | W | — | stays |

## 5. Assets and data

| Original path | Class | Destination | Status |
| --- | --- | --- | --- |
| `assets/images/figures/*.png` (123 files) | A | — | stays; article URLs must not change |
| `assets/images/headers/*` (83 files) | A | — | stays |
| `assets/data/gdp_q1_2025_release_vintages.csv` | R input | `data/economics/` (copy) | copied; website copy stays for the download link |

The GDP vintage CSV is transcribed from archived BEA releases and is linked from
the article as a download at `/assets/data/…`. The website copy stays so the link
keeps working; a copy travels here so the model is reproducible on its own.

---

## 6. Article-to-code map

Every published article backed by scientific code, with the figures it embeds.
Figure names are slugs under `/assets/images/figures/` on the website and under
`build/figures/<domain>/` here.

### Statistics

| Article | Permalink | Website source | Figures |
| --- | --- | --- | --- |
| `_posts/statistics/2026-09-12-confidence_sets_are_not_just_intervals.md` | `/statistics/confidence_sets_are_not_just_intervals/` | `generate_confidence_set_figures.py` | `confidence_set_components`, `confidence_set_profile_basins` |
| `_posts/science_communication/2024-09-12-why_a_million_responses_can_still_give_the_wrong_answer.md` | `/science-communication/why_a_million_responses_can_still_give_the_wrong_answer/` | `generate_poll_selection_figures.py`, `tests/test_poll_selection_models.py` | `science_poll_selection_precision`, `science_poll_selection_weighting` |
| `_posts/science_communication/2024-11-07-why_a_small_p_value_does_not_settle_a_claim.md` | `/science-communication/why_a_small_p_value_does_not_settle_a_claim/` | `generate_pvalue_evidence_figures.py`, `tests/test_pvalue_evidence_models.py` | `science_pvalue_selected_studies`, `science_pvalue_study_comparison` |
| `_posts/science_communication/2026-09-19-what_before_and_after_testimonials_can_establish.md` | `/science-communication/what_before_and_after_testimonials_can_establish/` | `generate_testimonial_figures.py`, `tests/test_testimonial_models.py` | `science_testimonial_selection`, `science_testimonial_counterfactual` |
| `_posts/statistics/2026-09-21-more_subjects_and_longer_trajectories.md` | `/statistics/more_subjects_and_longer_trajectories/` | `generate_2026_evidence_articles.py` (`longitudinal_budget`) | longitudinal design panels |

### Health and evidence communication

| Article | Permalink | Website source | Figures |
| --- | --- | --- | --- |
| `_posts/healthcare/2026-07-12-aspartame_fruit_true_premise_bad_argument.md` | `/healthcare/aspartame_fruit_true_premise_bad_argument/` | `generate_aspartame_dose_figures.py`, `tests/test_aspartame_dose_models.py` | `aspartame_methanol_sources`, `aspartame_blood_methanol` |
| `_posts/healthcare/2026-02-18-hormone_balance_social_media_myth.md` | `/healthcare/hormone_balance_social_media_myth/` | `generate_hormone_testing_figures.py`, `tests/test_hormone_testing_models.py` | `hormone_panel_false_flags`, `hormone_reference_change_values` |
| `_posts/healthcare/2026-09-02-inflammation_is_not_a_diagnosis_social_media_myths.md` | `/healthcare/inflammation_is_not_a_diagnosis_social_media_myths/` | `generate_inflammation_marker_figures.py`, `tests/test_inflammation_marker_models.py` | `inflammation_crp_scale`, `inflammation_trials_forest` |
| `_posts/healthcare/2026-06-03-leaky_gut_social_media_myths.md` | `/healthcare/leaky_gut_social_media_myths/` | `generate_leaky_gut_figures.py`, `tests/test_leaky_gut_models.py` | `leaky_gut_surrogate_test`, `leaky_gut_relatives_risk` |
| `_posts/healthcare/2026-03-05-consumer_microbiome_testing_limits.md` | `/healthcare/consumer_microbiome_testing_limits/` | `generate_microbiome_testing_figures.py`, `tests/test_microbiome_testing_models.py` | `microbiome_compositional_closure`, `microbiome_flag_repeatability` |
| `_posts/healthcare/2026-01-24-parasite_cleanse_social_media_myth.md` | `/healthcare/parasite_cleanse_social_media_myth/` | `generate_parasite_testing_figures.py`, `tests/test_parasite_testing_models.py` | `parasite_negative_results`, `parasite_repeat_sampling` |
| `_posts/healthcare/2026-05-11-results_are_not_evidence_influencer_science.md` | `/healthcare/results_are_not_evidence_influencer_science/` | `generate_results_rhetoric_figures.py`, `tests/test_results_rhetoric_models.py` | `results_rhetoric_retention`, `results_rhetoric_testimonial_wall` |
| `_posts/science_communication/2025-07-17-why_longer_survival_after_diagnosis_can_mislead.md` | `/science-communication/why_longer_survival_after_diagnosis_can_mislead/` | `generate_screening_survival_figures.py`, `tests/test_screening_survival_models.py` | `science_screening_survival_and_mortality`, `science_screening_duration_selection` |
| `_posts/healthcare/2026-09-28-what_a_wearable_heart_alert_can_tell_you.md` | `/healthcare/what_a_wearable_heart_alert_can_tell_you/` | `generate_2026_evidence_articles.py` (`wearable_alerts`) | wearable alert panels |

### Physics and science communication

| Article | Permalink | Website source | Figures |
| --- | --- | --- | --- |
| `_posts/science_communication/2025-01-23-quantum_measurement_and_the_observer_myth.md` | `/science-communication/quantum_measurement_and_the_observer_myth/` | `generate_quantum_observer_figures.py`, `tests/test_quantum_observer_models.py` | `science_quantum_marker_visibility`, `science_quantum_eraser_conditioning` |
| `_posts/science_communication/2024-02-15-cold_days_in_a_warming_climate.md` | `/science-communication/cold_days_in_a_warming_climate/` | `generate_science_communication_figures.py` | `science_weather_climate_shift`, `science_climate_kl_evidence` |
| `_posts/science_communication/2024-07-11-why_summer_follows_earths_tilt.md` | `/science-communication/why_summer_follows_earths_tilt/` | `generate_science_communication_figures.py` | `science_seasons_tilt_geometry` |
| `_posts/science_communication/2025-03-20-how_antibiotic_resistance_spreads.md` | `/science-communication/how_antibiotic_resistance_spreads/` | `generate_science_communication_figures.py` | `science_antibiotic_selection` |
| `_posts/science_communication/2025-10-09-randomness_does_not_owe_a_reversal.md` | `/science-communication/randomness_does_not_owe_a_reversal/` | `generate_science_communication_figures.py` | `science_streaks_three_mechanisms` |
| `_posts/science_communication/2026-02-12-natural_origin_does_not_establish_safety.md` | `/science-communication/natural_origin_does_not_establish_safety/` | `generate_science_communication_figures.py` | `science_concentration_and_amount` |
| `_posts/science_communication/2026-06-18-read_the_starting_risk_before_the_percentage.md` | `/science-communication/read_the_starting_risk_before_the_percentage/` | `generate_science_communication_figures.py` | `science_relative_absolute_risk` |
| `_posts/science_communication/2026-09-30-microwaves_heat_food_without_making_it_radioactive.md` | `/science-communication/microwaves_heat_food_without_making_it_radioactive/` | `generate_2026_evidence_articles.py` (`microwave_energy`) | microwave energy panels |
| `_posts/programming/2026-09-25-numerical_verification_before_optimization.md` | `/programming/numerical_verification_before_optimization/` | `generate_2026_evidence_articles.py` (`numerical_verification`) | convergence panels |

### Data engineering

| Article | Permalink | Website source | Figures |
| --- | --- | --- | --- |
| `_posts/data_science/2026-09-19-a_data_lake_is_a_directory_with_rules.md` | `/data-science/a_data_lake_is_a_directory_with_rules/` | `data_lake_benchmarks.py`, `generate_data_lake_figures.py`, `tests/test_data_lake_benchmarks.py` | `data_lake_formats_and_columns`, `data_lake_partition_layouts` |
| `_posts/data_science/2026-09-19-a_database_for_analysis_rows_columns_indexes.md` | `/data-science/a_database_for_analysis_rows_columns_indexes/` | `database_benchmarks.py`, `generate_database_figures.py`, `tests/test_database_benchmarks.py` | `database_rows_versus_columns`, `database_index_helps_and_hurts` |
| `_posts/machine_learning/2026-09-23-monitoring_without_labels_identifiability.md` | `/machine-learning/monitoring_without_labels_identifiability/` | `generate_2026_evidence_articles.py` (`monitoring_worlds`) | monitoring panels |
| `_posts/economics/2026-10-02-economic_data_have_two_dates.md` | `/economics/economic_data_have_two_dates/` | `generate_2026_evidence_articles.py` (`economic_vintages`), `assets/data/gdp_q1_2025_release_vintages.csv` | release timeline |

### Time series

| Article | Permalink | Website source | Figures |
| --- | --- | --- | --- |
| `_posts/data_science/2024-02-14-advanced_sequential_changepoint.md` | `/data-science/advanced_sequential_changepoint/` | removed — migration complete | `sequential_cusum_worked_example` |

### Drafts

`_drafts/ideas/` carries three models through
`generate_coverage_draft_figures.py`: an adaptive personal baseline
(`healthcare_adaptive_baseline`), randomisation balance
(`research_randomisation_balance`), and a storage ledger
(`environment_storage_constraints`). The drafts are unpublished, but the code is
scientific and its tests check real identities, so it migrates with the rest.
`_drafts/ideas/README.md` cites the old `assets/viz/` paths and has to be
rewritten as each group lands.

---

## 7. `generate_figures.py` — the one outstanding migration

Everything else in this inventory has moved. This file has not, and it is the
only scientific Python left in the website repository.

One 3,826-line module holds **76 figure generators** behind a `@figure(slug,
alt)` registry, all sharing a single module-level NumPy generator seeded at
20260816. Nothing but `README.md` refers to it by name; the articles refer to
the images it writes.

### What is in it

| Article subject | Figures | Examples |
| --- | ---: | --- |
| Statistics | 42 | selective inference, experiment design, bootstrap coverage, propensity estimators |
| Machine learning | 16 | drift monitoring and alerting, leakage, learning curves, permutation importance |
| Data science | 6 | kernel density bandwidth, splines, synthetic control, outlier plots, null-rate monitoring |
| Mathematics | 5 | central-limit convergence, Kaplan–Meier, MCMC, distance concentration, queueing |
| Economics | 3 | Lorenz/Gini, Solow steady state, Monte Carlo fan |
| Time series | 1 | penalised change-point partitioning |
| Referenced by no article | 3 | `acf_pacf`, `type_i_ii`, `linear_vs_logistic` |

### Progress

| Group | Figures | Status |
| --- | ---: | --- |
| Economics | 3 | **migrated** to `economics/inequality*.py`, `solow_growth*.py`, `gdp_fan*.py` |
| Mathematics, time series | 6 | **migrated** to `mathematics/` and `time_series/optimal_partition*.py` |
| Data science | 6 | **migrated** to `data_science/` |
| Machine learning | 16 | **migrated** to `machine_learning/` |
| Statistics | 42 | pending |

Migrated figures keep their website slugs, so a regenerated PNG replaces the
published one under the same name. Each takes its own seed, 20260816, the value
the shared generator used; its draws therefore differ from the published image,
which the tests accept because they check the model, not pixels. The exception
is a generator that already built its own seeded generator: the queueing,
change-point, and distance-concentration figures keep their draws exactly, and
their tests pin the numbers those articles print. Synthetic control does too.

One figure's claim does not survive migration. `splines_fit` is titled "a spline
bends locally; a high-degree polynomial wobbles globally", but with its design
(a smooth curve, 90 points, smoothing factor 0.09 per point) the degree-10
polynomial is closer to the true curve than the spline for 169 of 200 seeds,
at the edges as well as the interior. The figure is migrated unchanged and its
tests check only the fits' guarantees; the article needs a design that shows
the effect, such as a sharper feature or a higher degree, or a different claim. The website
copies of `generate_figures.py` generators stay until every group has moved, so
`website_cleanup` remains `pending` for these articles.

The last row is worth separating: `acf_pacf`, `type_i_ii`, and
`linear_vs_logistic` produce images that exist under `assets/images/figures/`
but that no current article embeds. `linear_vs_logistic` joined them after the
original survey, when its article stopped embedding the image; the
`null_monitor_detection` generator was added for a September 2026 article, which
keeps the total at 76. They should be confirmed dead before being migrated.

### What it would cost

| | |
| --- | --- |
| New dependencies | SciPy (22 figures), scikit-learn (6, now added), statsmodels (1) |
| Need none of those | 48 figures |
| Slow generators | the queueing figure simulates 200,000 events five times; the change-point figure runs an O(n²) dynamic program thirteen times |

### Why it is last, and how to split it

Every generator draws from one module-level generator in registration order, so
moving any figure out changes the draws every later figure receives. The
regenerated images will therefore not be byte-identical to the published ones.
That is acceptable — the published PNGs stay in the website repository and their
URLs do not change — but it means the split has to be deliberate rather than
incremental: each figure takes an explicit seed of its own as it moves.

The suggested order is smallest first, so the pattern and the dependency set are
settled before the large groups:

1. **Economics, mathematics, time series** (9 figures). Establishes the pattern
   and adds SciPy and scikit-learn. Several have real content to test: the Gini
   coefficient against the closed form for a lognormal, the Solow steady state
   as a fixed point, Kaplan–Meier against a hand-computed example.
2. **Data science** (6 figures).
3. **Machine learning** (16 figures) into a new `machine_learning/` submodule.
4. **Statistics** (42 figures), in three or four thematic groups.

Unlike the models already migrated, these figures are illustrations rather than
evidence: they draw synthetic data to make a point, and the articles do not
quote numbers from them. Their tests should therefore check determinism under a
fixed seed, the invariants each illustration depends on, and the derived
quantities that are worth stating — not published values, because there are
none.

---

## 8. Reference checks required before each cleanup

A cleanup pull request in the website repository has to leave no reference
behind. Search for, at minimum:

```text
assets/viz/generate_*.py
assets/viz/*_benchmarks.py
tests/test_*_models.py
tests/test_*_figures.py
tests/test_*_benchmarks.py
code/michelson_morley.py
code/Untitled.ipynb
```

across `_posts/`, `_drafts/`, `_pages/`, `docs/`, `_data/`, `README.md`,
`CONTRIBUTING.md`, `Rakefile`, and `.github/`. GitHub code search indexes the
default branch, so the check runs against the working branch with `grep`.
