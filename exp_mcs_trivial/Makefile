CSV ?= ../pdbbind_final_groups.csv
OUT ?= results
N_JOBS ?= 1

.PHONY: all mcs summary clean

all: mcs summary

mcs:
	python scripts/compute_mcs_trivial.py \
		--input-csv $(CSV) \
		--out-dir $(OUT) \
		--group-col final_group_id \
		--n-jobs $(N_JOBS)

summary:
	python scripts/summarize_mcs_trivial.py \
		--pairwise-csv $(OUT)/mcs_pairwise_quality.csv \
		--group-summary-csv $(OUT)/mcs_group_quality_summary.csv \
		--out-dir $(OUT)

clean:
	rm -rf $(OUT)
