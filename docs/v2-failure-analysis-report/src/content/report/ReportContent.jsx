import React from "react";

import {
  ChartRenderer,
  DataComponent,
  DataTable,
  MetricCard,
  ReportSection,
  RichNarrative,
  SortableItem,
  SortableRegion,
  useDataApp,
} from "../../data-app-public.jsx";

const failureChart = {
  type: "horizontalBar",
  x: "label",
  y: "tasks",
  colors: { tasks: "var(--chart-7)" },
  valueDecimals: 0,
  showXAxisLabel: false,
  showYAxisLabel: false,
};

const requestChart = {
  type: "horizontalBar",
  x: "status",
  y: "requests",
  colors: { requests: "var(--chart-1)" },
  valueDecimals: 0,
  showXAxisLabel: false,
  showYAxisLabel: false,
};

const sectionOrder = ["failure-breakdown", "request-breakdown", "task-audit", "report-methods"];

const taskColumns = [
  { field: "task", label: "Task", presentation: "identity", secondaryField: "stage" },
  { field: "result", label: "Result", presentation: "status" },
  { field: "diagnosis", label: "Primary diagnosis" },
  { field: "exact", label: "Exact outputs" },
  { field: "directGuarded", label: "Direct guarded" },
  { field: "synthGuarded", label: "Synth guarded" },
  { field: "requests", label: "Requests C/T/F" },
  { field: "sources", label: "Selected sources" },
  { field: "cellAccuracy", label: "Best cell", presentation: "percent" },
  { field: "oracle", label: "Oracle outputs" },
];

function pct(value, digits = 1) {
  return new Intl.NumberFormat(undefined, {
    style: "percent",
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value ?? 0);
}

function n(value) {
  return new Intl.NumberFormat().format(value ?? 0);
}

export function ReportContent() {
  const {
    reviewedRows,
    chartOverrides,
    chartProps,
    visible,
    canEdit,
    mode,
    appTitle,
    setAppTitle,
  } = useDataApp();
  const [summary] = reviewedRows("evaluation_summary");
  const failures = reviewedRows("failure_categories");
  const requests = reviewedRows("request_statuses");
  const taskRows = reviewedRows("task_audit");
  const failureSpec = chartOverrides["failure-breakdown"] ?? failureChart;
  const requestSpec = chartOverrides["request-breakdown"] ?? requestChart;
  const auditRows = taskRows.map((row) => ({
    task: row.task_id,
    stage: row.selection_stage,
    result: row.result,
    diagnosis: row.diagnosis,
    exact: `${row.correct_outputs}/${row.test_outputs}`,
    directGuarded: row.direct_guarded,
    synthGuarded: row.synthesis_guarded,
    requests: `${row.request_completed}/${row.request_timed_out}/${row.request_failed}`,
    sources: `${row.attempt_1_source} / ${row.attempt_2_source}`,
    cellAccuracy: row.best_selected_cell_accuracy,
    oracle: `${row.candidate_oracle_outputs}/${row.test_outputs}`,
  }));

  return <article className="report-content" aria-label="V2 public evaluation failure analysis">
    <header className="report-hero">
      <p className="report-kicker">ARC-AGI-2 · frozen V2 public evaluation</p>
      <h1 data-data-app-title contentEditable={canEdit && mode === "edit"} suppressContentEditableWarning
        aria-label={canEdit && mode === "edit" ? "Edit report heading" : undefined}
        onBlur={canEdit && mode === "edit" ? (event) => setAppTitle(event.currentTarget.textContent.trim() || appTitle) : undefined}
        onKeyDown={canEdit && mode === "edit" ? (event) => {
          if (event.key === "Enter") { event.preventDefault(); event.currentTarget.blur(); }
        } : undefined}>{appTitle}</h1>
      <RichNarrative id="report:description" className="report-deck"
        label="Edit report introduction"
        value="A post-freeze audit of all **120 public tasks**. The result is decisive: V2 mostly measured timeout handling and unguarded retrieval—not the intended Luna resynthesis strategy." />
    </header>

    {visible("report-summary") && <ReportSection id="report-summary" title="The score reflects a coverage failure"
      queryId="evaluation_summary" sourceRows={[summary]} showHeading={false} className="report-summary">
      <RichNarrative id="report-summary:body" className="report-summary-lead" label="Edit executive finding"
        value={`## The score reflects a coverage failure

V2 solved **${summary?.correct_tasks}/${summary?.tasks} tasks (${pct(summary?.strict_task_accuracy)})** and **${summary?.correct_outputs}/${summary?.test_outputs} outputs (${pct(summary?.pass_at_2)} Pass@2)**. Yet **${summary?.model_requests_timed_out}/${summary?.model_requests} model requests (${pct(summary?.model_timeout_rate)}) timed out**, and only **${summary?.tasks_with_guarded_candidate}/${summary?.tasks} tasks** received any guard-passing candidate.

All three solved tasks used guard-passing programs. No failed task contained a stored candidate that could exactly solve all its labelled outputs, so this run shows **no evidence that final candidate selection was the primary bottleneck**.`} />
    </ReportSection>}

    <div className="report-facts" aria-label="Key evaluation metrics">
      {visible("metric-strict") && <MetricCard id="metric-strict" title="Strict accuracy"
        queryId="evaluation_summary" sourceRows={[summary]} value={pct(summary?.strict_task_accuracy)}
        description={`${summary?.correct_tasks} of ${summary?.tasks} tasks solved exactly.`} />}
      {visible("metric-pass-at-2") && <MetricCard id="metric-pass-at-2" title="Pass@2"
        queryId="evaluation_summary" sourceRows={[summary]} value={pct(summary?.pass_at_2)}
        description={`${summary?.correct_outputs} of ${summary?.test_outputs} labelled outputs solved.`} />}
      {visible("metric-timeout") && <MetricCard id="metric-timeout" title="Request timeout rate"
        queryId="evaluation_summary" sourceRows={[summary]} value={pct(summary?.model_timeout_rate)}
        negative description={`${summary?.model_requests_timed_out} of ${summary?.model_requests} model requests exceeded 360 seconds.`} />}
      {visible("metric-coverage") && <MetricCard id="metric-coverage" title="Guarded task coverage"
        queryId="evaluation_summary" sourceRows={[summary]} value={pct(summary?.guarded_task_coverage)}
        description={`${summary?.tasks_with_guarded_candidate} of ${summary?.tasks} tasks obtained a fully guarded program.`} />}
    </div>

    {visible("pipeline-funnel") && <DataComponent id="pipeline-funnel" title="What actually reached scoring"
      queryId="evaluation_summary" kind="custom" sourceRows={[summary]} displayRows={[summary]}
      description="Direct retrieval and model resynthesis are parallel paths; counts use their natural task, candidate, or request grain.">
      <div className="pipeline-paths" data-reviewed-rows>
        <section className="pipeline-path" aria-label="Direct retrieval path">
          <span className="path-label">Direct retrieval</span>
          <div className="path-steps">
            <div><strong>{n(summary?.tasks)}</strong><span>tasks</span></div>
            <span className="path-arrow" aria-hidden="true">→</span>
            <div><strong>{n(summary?.direct_candidates_tested)}</strong><span>program checks</span></div>
            <span className="path-arrow" aria-hidden="true">→</span>
            <div className="path-emphasis"><strong>{n(summary?.direct_guarded_candidates)}</strong><span>guard pass</span></div>
          </div>
        </section>
        <section className="pipeline-path" aria-label="Model resynthesis path">
          <span className="path-label">Luna resynthesis</span>
          <div className="path-steps">
            <div><strong>{n(summary?.model_requests)}</strong><span>requests</span></div>
            <span className="path-arrow" aria-hidden="true">→</span>
            <div><strong>{n(summary?.model_requests_completed)}</strong><span>completed</span></div>
            <span className="path-arrow" aria-hidden="true">→</span>
            <div><strong>{n(summary?.synthesis_candidates)}</strong><span>parsed programs</span></div>
            <span className="path-arrow" aria-hidden="true">→</span>
            <div className="path-emphasis"><strong>{n(summary?.synthesis_guarded_candidates)}</strong><span>guard passes</span></div>
          </div>
        </section>
      </div>
    </DataComponent>}

    <SortableRegion id="report:sections" label="Report sections" variant="stack"
      authoredOrder={sectionOrder} className="report-sortable-sections">
      {visible("failure-breakdown") && <SortableItem id="failure-breakdown" label="Failure breakdown" kind="chart">
        <section className="report-section">
          <ReportSection id="failure-interpretation" title="115 tasks ended in timeout-driven fallback"
            queryId="failure_categories" sourceRows={failures} showHeading={false}>
            <RichNarrative id="failure-interpretation:body" className="report-analysis" label="Edit failure interpretation"
              value={`## 115 tasks ended in timeout-driven fallback

The six-minute turn limit was the dominant terminal condition. Two more tasks received completed synthesis results that failed the deterministic guards. The only three tasks with selected guard-passing candidates were the three exact successes.`} />
          </ReportSection>
          <DataComponent id="failure-breakdown" title="Outcome and primary operational diagnosis"
            queryId="failure_categories" kind="chart" chart={failureSpec}
            displayRows={failures} sourceRows={failures}
            description="Mutually exclusive task counts; each task is assigned one observable primary diagnosis.">
            <ChartRenderer spec={failureSpec} rows={failures} height={250} {...chartProps("failure-breakdown")} />
          </DataComponent>
        </section>
      </SortableItem>}

      {visible("request-breakdown") && <SortableItem id="request-breakdown" label="Request reliability" kind="chart">
        <section className="report-section">
          <ReportSection id="request-interpretation" title="The model path completed on only four tasks"
            queryId="request_statuses" sourceRows={requests} showHeading={false}>
            <RichNarrative id="request-interpretation:body" className="report-analysis" label="Edit request interpretation"
              value={`## The model path completed on only four tasks

There were **${summary?.model_requests_completed} completed requests**, producing ${summary?.synthesis_candidates} parsed programs across ${summary?.tasks_with_completed_model} tasks. Two programs passed every guard and both solved their hidden output. The post-hoc candidate oracle is also ${summary?.candidate_oracle_correct_outputs}/${summary?.test_outputs}, exactly equal to the submitted score—there was no latent correct program waiting to be selected.`} />
          </ReportSection>
          <DataComponent id="request-breakdown" title="Durable model-request outcomes"
            queryId="request_statuses" kind="chart" chart={requestSpec}
            displayRows={requests} sourceRows={requests}
            description="121 request records: completed, timed out at 360 seconds, or failed.">
            <ChartRenderer spec={requestSpec} rows={requests} height={230} {...chartProps("request-breakdown")} />
          </DataComponent>
        </section>
      </SortableItem>}

      {visible("task-audit") && <SortableItem id="task-audit" label="All 120 tasks" kind="table">
        <section className="report-section">
          <ReportSection id="task-audit-intro" title="Every task is traceable to its terminal condition"
            queryId="task_audit" sourceRows={taskRows} showHeading={false}>
            <RichNarrative id="task-audit-intro:body" className="report-analysis" label="Edit task-audit introduction"
              value={`## Every task is traceable to its terminal condition

Search or sort the complete 120-task audit. \`C/T/F\` means completed, timed-out, and failed model requests. \`Oracle outputs\` counts labels matched by any stored candidate after freeze; it is diagnostic only and cannot be used by a deployable selector.`} />
          </ReportSection>
          <DataComponent id="task-audit" title="Per-task failure audit" queryId="task_audit" kind="table"
            sourceRows={taskRows} displayRows={auditRows}
            description="Exact scoring, guard coverage, request outcomes, selected sources, partial cell accuracy, and post-hoc candidate coverage.">
            <DataTable rows={auditRows} columns={taskColumns} rowKey="task"
              caption="All 120 ARC-AGI-2 public evaluation tasks" compactNumbers={false} />
          </DataComponent>
        </section>
      </SortableItem>}

      {visible("report-methods") && <SortableItem id="report-methods" label="What to change next" kind="narrative">
        <ReportSection id="report-methods" title="V3 should gate response ingestion before another full run"
          queryId="evaluation_summary" queryIds={["evaluation_summary", "request_statuses", "task_audit"]}
          sourceRowsByQuery={{ evaluation_summary: [summary], request_statuses: requests, task_audit: taskRows }}
          showHeading={false} className="report-methods">
          <RichNarrative id="report-methods:body" className="report-caveat" label="Edit recommendations and limitations"
            value={`## V3 should gate response ingestion before another full run

1. Run a 10-task held-out smoke test and require at least nine usable model responses before the deadline. Shorten prompts or output limits, or reduce reasoning effort; simply raising the timeout at the same concurrency reduces 120-task coverage.
2. Preserve timed-out background response IDs and ingest late completions exactly once while the global budget remains, instead of treating 360 seconds as a terminal candidate-generation failure.
3. Rework retrieval around task-family and behavioral features. ${n(summary?.direct_candidates_tested)} direct checks produced only one fully compatible public task.
4. Test selector changes on held-out training tasks. Public labels in this report are post-hoc evidence, not allowable inference features.
5. Keep this frozen run as V2 and execute V3 in a new workspace under the same 12-hour active-time protocol.

### Method and limitation

This report joins the frozen submission and labelled public files to read-only SQLite candidate, verifier, request, and provenance rows. No candidate was rerun and no frozen artifact was modified. A timeout diagnosis cannot establish what a still-running model response would eventually have produced.`} />
        </ReportSection>
      </SortableItem>}
    </SortableRegion>
  </article>;
}
