UV ?= $(shell command -v uv 2>/dev/null || printf '%s/.local/bin/uv' "$$HOME")

.PHONY: sync run task t01 monitor monitor-task classify analytics report skills generalization-check snapshot-skills evolution clean

sync:
	$(UV) sync

run:
	$(UV) run python main.py

task:
	@if [ -z "$(TASKS)" ]; then echo "usage: make task TASKS='t01 t03'"; exit 1; fi
	$(UV) run python main.py $(TASKS)

t01:
	$(UV) run python main.py t01

monitor:
	$(UV) run python main.py --monitor --classify-tasks

monitor-task:
	@if [ -z "$(TASKS)" ]; then echo "usage: make monitor-task TASKS='t01 t03'"; exit 1; fi
	$(UV) run python main.py --monitor --classify-tasks $(TASKS)

classify:
	$(UV) run python main.py --monitor --classify-tasks --classify-only

analytics:
	$(UV) run python analytics_cli.py summary

report:
	$(UV) run python analytics_cli.py report

skills:
	$(UV) run python analytics_cli.py skills

generalization-check:
	$(UV) run python analytics_cli.py check-generalization

snapshot-skills:
	$(UV) run python analytics_cli.py snapshot-skills

evolution:
	$(UV) run python analytics_cli.py evolution

clean:
	rm -rf .venv
