const STORAGE_KEY = 'supernpubench-plan-board-v2';
const LEGACY_STORAGE_KEY = 'supernpubench-plan-board-v1';
let repositoryData;
let boardData;
let draftData;
let activeStatus = 'all';
let activeEditorSection = 'meta';
let inlineEditingTaskIndex = null;
let inlineTaskDraft = null;
let draggedTaskIndex = null;
const dirtyTasks = new Set();

const statusLabels = {
  in_progress: '进行中', blocked: '问题阻塞', planned: '待开始',
  partial: '部分通过', pass: '通过', done: '已完成', released: '已发布'
};

const editorSections = {
  meta: {
    label: '基本信息', description: '标题、负责人、更新时间和当前重点。', singleton: true,
    fields: [
      ['title', '看板标题', 'text'], ['subtitle', '副标题', 'text'], ['owner', '负责人 / 团队', 'text'],
      ['lastUpdated', '最后更新', 'date'], ['currentFocus', '当前重点', 'textarea']
    ]
  },
  summary: {
    label: '摘要指标', description: '首页顶部的关键数字卡片。',
    fields: [['label', '指标名称', 'text'], ['value', '数值', 'text'], ['detail', '补充说明', 'text'], ['tone', '颜色', 'select', ['green', 'blue', 'orange', 'purple']]],
    create: () => ({ label: '新指标', value: '0', detail: '待更新', tone: 'blue' })
  },
  released: {
    label: '已发布特性', description: '维护已经完成并发布的功能、版本和验证结果。',
    fields: [['date', '发布日期', 'date'], ['title', '特性名称', 'text'], ['version', '版本 / 分支', 'text'], ['description', '功能描述', 'textarea'], ['result', '发布结果', 'textarea'], ['tags', '标签（逗号分隔）', 'list']],
    create: () => ({ date: today(), title: '新发布特性', version: 'main', description: '', result: '', tags: [] })
  },
  currentWork: {
    label: '当前工作', description: '维护任务状态、阶段、当前问题和下一步。',
    fields: [['id', '任务 ID', 'text'], ['title', '任务名称', 'text'], ['area', '所属领域', 'text'], ['status', '状态', 'select', ['in_progress', 'blocked', 'planned', 'done']], ['priority', '优先级', 'select', ['P0', 'P1', 'P2', 'P3']], ['description', '工作说明', 'textarea'], ['stages', '阶段状态', 'stages'], ['currentIssueSummary', '当前问题总结', 'textarea'], ['next', '下一步', 'textarea']],
    create: () => ({ id: `WORK-${Date.now().toString().slice(-4)}`, title: '新工作项', area: 'Operator', status: 'planned', priority: 'P1', description: '点击此处填写工作说明。', stages: defaultStages(), currentIssueSummary: '点击此处填写当前问题总结。', next: '点击此处填写下一步。' })
  },
  plans: {
    label: '版本发布计划', description: '维护版本号、发布时间、发布内容、特性和预计增加的算子。',
    fields: [['id', '计划 ID', 'text'], ['version', '版本号', 'text'], ['target', '计划发布日期', 'date'], ['status', '状态', 'select', ['planned', 'in_progress', 'blocked', 'done']], ['priority', '优先级', 'select', ['P0', 'P1', 'P2', 'P3']], ['title', '版本定位', 'text'], ['description', '计划发布内容', 'textarea'], ['features', '本版本特性（逗号分隔）', 'list'], ['operators', '预计增加的算子', 'operatorChecklist']],
    create: () => ({ id: `PLAN-${Date.now().toString().slice(-4)}`, version: 'release_verXXXX', target: today(), status: 'planned', priority: 'P1', title: '新版本', description: '', features: [], operators: [] })
  },
  results: {
    label: '验证结果', description: '记录功能验证、精度校验和性能分析结论。',
    fields: [['date', '验证日期', 'date'], ['title', '结果名称', 'text'], ['status', '状态', 'select', ['pass', 'partial', 'blocked', 'done']], ['value', '核心数据', 'text'], ['summary', '结果摘要', 'textarea']],
    create: () => ({ date: today(), title: '新验证结果', status: 'pass', value: '-', summary: '' })
  },
  versions: {
    label: '版本基线', description: '维护仓库、分支、Commit 和补充说明。',
    fields: [['name', '仓库 / 组件', 'text'], ['branch', '分支', 'text'], ['commit', 'Commit ID', 'text'], ['note', '说明', 'textarea']],
    create: () => ({ name: '新组件', branch: 'main', commit: '', note: '' })
  }
};
const editorSectionOrder = ['meta', 'summary', 'currentWork', 'plans', 'results', 'versions', 'released'];

const escapeHtml = (value = '') => String(value)
  .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
  .replaceAll('"', '&quot;').replaceAll("'", '&#039;');

function clone(data) { return JSON.parse(JSON.stringify(data)); }
function today() { return new Date().toISOString().slice(0, 10); }
function defaultStages() {
  return ['ISA', '算子', '编译器', 'gfrun'].map(name => ({ name, status: 'planned' }));
}
function normalizeStageStatus(status) {
  if (status === 'pass') return 'done';
  if (status === 'partial') return 'in_progress';
  return ['planned', 'in_progress', 'done', 'blocked'].includes(status) ? status : 'planned';
}

function validateData(data) {
  const arrays = ['summary', 'released', 'currentWork', 'plans', 'results', 'versions'];
  if (!data || typeof data.meta !== 'object') throw new Error('缺少基本信息 meta。');
  arrays.forEach(key => { if (!Array.isArray(data[key])) throw new Error(`缺少列表 ${key}。`); });
}

function migrateLocalData(localData) {
  const migrated = clone(localData);
  migrated.meta.dataRevision = repositoryData.meta.dataRevision;
  migrated.currentWork.forEach(item => {
    const repositoryItem = repositoryData.currentWork.find(candidate => candidate.id === item.id);
    const stages = item.stages?.length ? item.stages : repositoryItem?.stages || defaultStages();
    item.stages = stages.map(stage => ({ name: stage.name, status: normalizeStageStatus(stage.status) }));
    if (!item.currentIssueSummary) item.currentIssueSummary = repositoryItem?.currentIssueSummary || '';
    delete item.progress;
  });
  migrated.plans = migrated.plans.map(item => {
    const repositoryItem = repositoryData.plans.find(candidate => candidate.id === item.id);
    const next = { ...item };
    next.version = repositoryItem?.version || item.version || item.milestone || 'release_verXXXX';
    next.features = clone(repositoryItem?.features || item.features || item.deliverables || []);
    const operators = item.operators || repositoryItem?.operators || [];
    next.operators = operators.map(operator => typeof operator === 'string'
      ? { name: operator, done: false }
      : { name: operator.name || '', done: Boolean(operator.done) });
    delete next.milestone;
    delete next.deliverables;
    return next;
  });
  repositoryData.released.slice().reverse().forEach(item => {
    if (!migrated.released.some(candidate => candidate.title === item.title)) migrated.released.unshift(clone(item));
  });
  const releasedSummary = migrated.summary.find(item => item.label === '已发布特性');
  const repositoryReleasedSummary = repositoryData.summary.find(item => item.label === '已发布特性');
  if (releasedSummary && repositoryReleasedSummary) Object.assign(releasedSummary, repositoryReleasedSummary);
  return migrated;
}

async function loadData() {
  const response = await fetch('./plan_data.json', { cache: 'no-store' });
  if (!response.ok) throw new Error(`无法加载 plan_data.json (${response.status})`);
  repositoryData = await response.json();
  validateData(repositoryData);
  try {
    const local = localStorage.getItem(STORAGE_KEY) || localStorage.getItem(LEGACY_STORAGE_KEY);
    const localData = local ? JSON.parse(local) : null;
    const repositoryRevision = Number(repositoryData.meta.dataRevision || 0);
    const localRevision = Number(localData?.meta?.dataRevision || 0);
    boardData = localData
      ? (localRevision >= repositoryRevision ? localData : migrateLocalData(localData))
      : clone(repositoryData);
    validateData(boardData);
    if (local || localRevision < repositoryRevision) localStorage.setItem(STORAGE_KEY, JSON.stringify(boardData));
  } catch (error) {
    console.warn('本地数据无效，已恢复仓库版本。', error);
    boardData = clone(repositoryData);
    localStorage.removeItem(STORAGE_KEY);
    localStorage.removeItem(LEGACY_STORAGE_KEY);
  }
  render();
}

function render() {
  document.querySelector('#boardTitle').textContent = boardData.meta.title;
  document.querySelector('#boardSubtitle').textContent = boardData.meta.subtitle;
  document.querySelector('#currentFocus').textContent = boardData.meta.currentFocus;
  document.querySelector('#lastUpdated').textContent = boardData.meta.lastUpdated;
  document.querySelector('#owner').textContent = boardData.meta.owner;
  document.title = boardData.meta.title;
  renderSummary(); renderReleased(); renderCurrentWork(); renderPlans(); renderResults(); renderVersions();
}

function renderSummary() {
  document.querySelector('#summaryGrid').innerHTML = boardData.summary.map(item => `
    <article class="summary-card ${escapeHtml(item.tone)}"><span class="label">${escapeHtml(item.label)}</span><strong>${escapeHtml(item.value)}</strong><small>${escapeHtml(item.detail)}</small></article>`).join('');
}

function renderReleased() {
  document.querySelector('#releasedList').innerHTML = boardData.released.map((item, index) => `
    <article class="release-item"><button type="button" class="card-edit" data-edit-record="released" data-index="${index}" aria-label="编辑发布特性 ${escapeHtml(item.title)}">编辑</button><time class="release-date">${escapeHtml(item.date)}</time><div class="release-content"><h3>${escapeHtml(item.title)} <span class="tag">${escapeHtml(item.version)}</span></h3><p>${escapeHtml(item.description)}</p><p class="release-result">结果：${escapeHtml(item.result)}</p></div><div class="tag-list">${(item.tags || []).map(tag => `<span class="tag">${escapeHtml(tag)}</span>`).join('')}</div></article>`).join('') || emptyState('还没有发布记录。');
}

function renderCurrentWork() {
  const items = activeStatus === 'all' ? boardData.currentWork : boardData.currentWork.filter(item => item.status === activeStatus);
  const cards = items.map(item => {
    const index = boardData.currentWork.indexOf(item);
    const stages = item.stages?.length ? item.stages : defaultStages();
    const dirty = dirtyTasks.has(index);
    return `<article class="work-card ${dirty ? 'is-dirty' : ''}" data-task-card="${index}">
      <div class="work-top"><div class="work-identity"><button type="button" class="drag-handle" draggable="true" data-drag-task="${index}" aria-label="拖动调整 ${escapeHtml(item.title)} 的顺序" title="拖动排序">⠿</button><span class="work-id editable-text" contenteditable="plaintext-only" data-card-field="id" role="textbox">${escapeHtml(item.id)}</span><span class="editable-text area-text" contenteditable="plaintext-only" data-card-field="area" role="textbox">${escapeHtml(item.area)}</span></div><div class="work-config"><select data-card-select="status" aria-label="任务状态">${renderTaskStatusOptions(item.status)}</select><select data-card-select="priority" aria-label="任务优先级">${renderPriorityOptions(item.priority)}</select><button type="button" class="mini-button delete" data-delete-task="${index}">删除</button></div></div>
      <h3 class="editable-text" contenteditable="plaintext-only" data-card-field="title" role="textbox">${escapeHtml(item.title)}</h3>
      ${renderEditableBlock('工作说明', 'description', item.description)}
      <div class="stage-block"><div class="stage-head"><b>阶段状态</b><button type="button" class="add-stage-button" data-card-add-stage="${index}">＋ 阶段</button></div><div class="stage-list">${stages.map((stage, stageIndex) => renderStage(stage, index, stageIndex)).join('')}</div></div>
      ${renderEditableBlock('当前问题总结', 'currentIssueSummary', item.currentIssueSummary)}
      ${renderEditableBlock('下一步', 'next', item.next)}
      <div class="card-save-bar"><span class="save-state">${dirty ? '有尚未保存的修改' : '内容已保存'}</span><button type="button" class="save-card-button ${dirty ? 'needs-save' : ''}" data-save-task="${index}">${dirty ? '保存修改' : '已保存'}</button></div>
    </article>`;
  }).join('');
  document.querySelector('#currentWorkList').innerHTML = `${cards}<button type="button" class="work-card-add" data-add-work-card aria-label="快速添加工作卡"><span>＋</span><b>添加工作卡</b><small>${items.length ? '在列表末尾创建' : '当前筛选下暂无卡片'}</small></button>`;
}

function renderStage(stage, taskIndex, stageIndex) {
  return `<div class="stage-item stage-${escapeHtml(stage.status)}" data-stage-index="${stageIndex}"><span class="stage-name editable-text" contenteditable="plaintext-only" data-stage-name role="textbox">${escapeHtml(stage.name)}</span><select class="stage-status-select ${escapeHtml(stage.status)}" data-card-stage-status aria-label="${escapeHtml(stage.name)} 阶段状态">${renderStageStatusOptions(stage.status)}</select><button type="button" class="stage-remove" data-card-delete-stage aria-label="删除 ${escapeHtml(stage.name)} 阶段">×</button></div>`;
}

function renderEditableBlock(label, field, value) {
  return `<div class="card-text-block"><b>${escapeHtml(label)}</b><p class="editable-text" contenteditable="plaintext-only" data-card-field="${field}" role="textbox">${escapeHtml(value || '')}</p></div>`;
}

function renderTaskStatusOptions(value) {
  return ['planned', 'in_progress', 'blocked', 'done'].map(option => `<option value="${option}" ${option === value ? 'selected' : ''}>${statusLabels[option]}</option>`).join('');
}

function renderPriorityOptions(value) {
  return ['P0', 'P1', 'P2', 'P3'].map(option => `<option value="${option}" ${option === value ? 'selected' : ''}>${option}</option>`).join('');
}

function renderStageStatusOptions(value) {
  return ['planned', 'in_progress', 'done', 'blocked'].map(option => `<option value="${option}" ${option === value ? 'selected' : ''}>${statusLabels[option]}</option>`).join('');
}

function renderInlineTaskEditor(index) {
  const item = inlineTaskDraft;
  return `<article class="work-card inline-editing" data-inline-task-index="${index}">
    <div class="inline-editor-head"><div><span>INLINE EDIT</span><b>直接编辑任务卡片</b></div><div><button type="button" class="quiet" data-inline-cancel>取消</button><button type="button" class="primary" data-inline-save>保存卡片</button></div></div>
    <div class="inline-form-grid">
      ${renderInlineField('id', '任务 ID', item.id)}${renderInlineField('title', '任务名称', item.title)}${renderInlineField('area', '所属领域', item.area)}
      ${renderInlineSelect('status', '状态', item.status, ['in_progress', 'blocked', 'planned', 'done'])}${renderInlineSelect('priority', '优先级', item.priority, ['P0', 'P1', 'P2', 'P3'])}
      ${renderInlineField('description', '工作说明', item.description, 'textarea', true)}
    </div>
    <div class="inline-stage-editor"><div class="inline-subhead"><div><b>阶段状态</b><span>默认包含 ISA、算子、编译器和 gfrun，也可以继续添加阶段。</span></div><button type="button" class="quiet" data-inline-add-stage>＋ 添加阶段</button></div><div class="stage-edit-list">${(item.stages || []).map(renderInlineStage).join('')}</div></div>
    <div class="inline-form-grid">${renderInlineField('currentIssueSummary', '当前问题总结', item.currentIssueSummary, 'textarea', true)}${renderInlineField('next', '下一步', item.next, 'textarea', true)}</div>
  </article>`;
}

function renderInlineField(name, label, value, type = 'text', wide = false) {
  const control = type === 'textarea'
    ? `<textarea rows="3" data-inline-field="${name}">${escapeHtml(value ?? '')}</textarea>`
    : `<input type="${type}" ${type === 'number' ? 'min="0" max="100"' : ''} value="${escapeHtml(value ?? '')}" data-inline-field="${name}" />`;
  return `<label class="inline-field ${wide ? 'wide' : ''}"><span>${escapeHtml(label)}</span>${control}</label>`;
}

function renderInlineSelect(name, label, value, options) {
  return `<label class="inline-field"><span>${escapeHtml(label)}</span><select data-inline-field="${name}">${options.map(option => `<option value="${escapeHtml(option)}" ${option === value ? 'selected' : ''}>${escapeHtml(statusLabels[option] || option)}</option>`).join('')}</select></label>`;
}

function renderInlineStage(stage, stageIndex) {
  return `<div class="stage-edit-row" data-inline-stage-index="${stageIndex}"><input value="${escapeHtml(stage.name)}" data-inline-stage-field="name" aria-label="阶段名称" /><select data-inline-stage-field="status" aria-label="阶段状态">${renderStageStatusOptions(stage.status)}</select><button type="button" class="mini-button delete" data-inline-delete-stage="${stageIndex}">删除</button></div>`;
}

function renderPlans() {
  document.querySelector('#planList').innerHTML = boardData.plans.map((item, index) => `
    <article class="plan-item"><button type="button" class="card-edit" data-edit-record="plans" data-index="${index}" aria-label="编辑版本 ${escapeHtml(item.version)}">编辑</button>
      <div class="plan-release-column">
        <div class="plan-card-head"><div><span class="plan-version">${escapeHtml(item.version)}</span><span class="work-id">${escapeHtml(item.id)}</span></div><div class="plan-state"><span class="status ${escapeHtml(item.status)}">${statusLabels[item.status] || escapeHtml(item.status)}</span><b>${escapeHtml(item.priority)}</b></div></div>
        <div class="plan-release-box"><div class="plan-title"><div><small>计划发布 · ${escapeHtml(item.target)}</small><h3>${escapeHtml(item.title)}</h3></div></div><p>${escapeHtml(item.description)}</p><b class="feature-label">本版本包含的特性</b><div class="version-features">${(item.features || []).map(value => `<span>${escapeHtml(value)}</span>`).join('') || '<em>暂未填写特性</em>'}</div></div>
      </div>
      <div class="plan-operators-column"><div class="operator-column-head"><div><small>OPERATOR CHECKLIST</small><b>预计增加的算子列表</b></div><span>${(item.operators || []).filter(operator => operator.done).length} / ${(item.operators || []).length}</span></div><div class="operator-checklist">${(item.operators || []).map((operator, operatorIndex) => `<label class="operator-check ${operator.done ? 'is-done' : ''}"><input type="checkbox" data-plan-operator-toggle data-plan-index="${index}" data-operator-index="${operatorIndex}" ${operator.done ? 'checked' : ''}><span>${escapeHtml(operator.name)}</span></label>`).join('') || '<p class="operator-empty">尚未添加预计支持的算子，可点击“编辑”补充。</p>'}</div></div>
    </article>`).join('') || emptyState('还没有版本发布计划。');
}

function renderResults() {
  document.querySelector('#resultList').innerHTML = boardData.results.map((item, index) => `
    <article class="result-card ${escapeHtml(item.status)}"><button type="button" class="card-edit" data-edit-record="results" data-index="${index}" aria-label="编辑结果 ${escapeHtml(item.title)}">编辑</button><div class="result-top"><time>${escapeHtml(item.date)}</time><span class="status ${escapeHtml(item.status)}">${statusLabels[item.status] || escapeHtml(item.status)}</span></div><h3>${escapeHtml(item.title)}</h3><strong>${escapeHtml(item.value)}</strong><p>${escapeHtml(item.summary)}</p></article>`).join('') || emptyState('还没有验证结果。');
}

function renderVersions() {
  document.querySelector('#versionList').innerHTML = boardData.versions.map(item => `
    <article><b>${escapeHtml(item.name)}</b><code>${escapeHtml(item.branch)}</code><code>${escapeHtml(item.commit)}</code><span>${escapeHtml(item.note)}</span></article>`).join('') || emptyState('还没有版本基线。');
}

function emptyState(message) { return `<p class="empty">${escapeHtml(message)}</p>`; }

function markTaskDirty(index, card) {
  dirtyTasks.add(index);
  card.classList.add('is-dirty');
  const button = card.querySelector('[data-save-task]');
  const state = card.querySelector('.save-state');
  if (button) { button.textContent = '保存修改'; button.classList.add('needs-save'); }
  if (state) state.textContent = '有尚未保存的修改';
}

function addWorkCard() {
  boardData.currentWork.push(editorSections.currentWork.create());
  const index = boardData.currentWork.length - 1;
  dirtyTasks.add(index);
  activeStatus = 'all';
  document.querySelectorAll('#statusFilters button').forEach(button => button.classList.toggle('active', button.dataset.status === 'all'));
  renderCurrentWork();
  requestAnimationFrame(() => document.querySelector(`[data-task-card="${index}"]`)?.scrollIntoView({ behavior: 'smooth', block: 'center' }));
}

function saveWorkCard(index) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(boardData));
  dirtyTasks.delete(index);
  renderCurrentWork();
}

function deleteWorkCard(index) {
  if (!confirm(`确认删除工作卡“${boardData.currentWork[index].title}”？`)) return;
  boardData.currentWork.splice(index, 1);
  dirtyTasks.clear();
  localStorage.setItem(STORAGE_KEY, JSON.stringify(boardData));
  renderCurrentWork();
}

function reorderWorkCard(fromIndex, toIndex) {
  if (fromIndex === toIndex || fromIndex < 0 || toIndex < 0) return;
  const [moved] = boardData.currentWork.splice(fromIndex, 1);
  boardData.currentWork.splice(toIndex, 0, moved);
  dirtyTasks.clear();
  localStorage.setItem(STORAGE_KEY, JSON.stringify(boardData));
  renderCurrentWork();
}

function startInlineTaskEdit(index) {
  inlineEditingTaskIndex = index;
  inlineTaskDraft = clone(boardData.currentWork[index]);
  if (!inlineTaskDraft.stages?.length) {
    inlineTaskDraft.stages = defaultStages();
  }
  if (!inlineTaskDraft.currentIssueSummary) inlineTaskDraft.currentIssueSummary = inlineTaskDraft.result || '';
  renderCurrentWork();
  requestAnimationFrame(() => document.querySelector('.work-card.inline-editing')?.scrollIntoView({ behavior: 'smooth', block: 'center' }));
}

function cancelInlineTaskEdit() {
  inlineEditingTaskIndex = null;
  inlineTaskDraft = null;
  renderCurrentWork();
}

function saveInlineTaskEdit() {
  boardData.currentWork[inlineEditingTaskIndex] = clone(inlineTaskDraft);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(boardData));
  inlineEditingTaskIndex = null;
  inlineTaskDraft = null;
  renderCurrentWork();
}

function updateInlineTaskControl(control) {
  const field = control.dataset.inlineField;
  if (!field) return;
  inlineTaskDraft[field] = control.value;
}

function updateInlineStageControl(control) {
  const row = control.closest('[data-inline-stage-index]');
  if (!row) return;
  inlineTaskDraft.stages[Number(row.dataset.inlineStageIndex)][control.dataset.inlineStageField] = control.value;
}

function openEditor(section = 'meta', recordIndex = null) {
  draftData = clone(boardData);
  activeEditorSection = editorSections[section] ? section : 'meta';
  document.querySelector('#editorMessage').textContent = '';
  renderEditor();
  document.querySelector('#editorDialog').showModal();
  if (recordIndex !== null) requestAnimationFrame(() => {
    const card = document.querySelector(`.form-card[data-record-index="${recordIndex}"]`);
    if (!card) return;
    card.scrollIntoView({ block: 'center' });
    card.classList.add('targeted');
    card.querySelector('input, textarea, select')?.focus({ preventScroll: true });
  });
}

function renderEditor() {
  const config = editorSections[activeEditorSection];
  document.querySelector('#editorNav').innerHTML = editorSectionOrder.map(key => {
    const item = editorSections[key];
    const count = item.singleton ? '' : `<span>${draftData[key].length}</span>`;
    return `<button type="button" class="${key === activeEditorSection ? 'active' : ''}" data-editor-nav="${key}">${escapeHtml(item.label)}${count}</button>`;
  }).join('');
  document.querySelector('#editorSectionTitle').textContent = config.label;
  document.querySelector('#editorSectionDescription').textContent = config.description;
  document.querySelector('#addItemButton').hidden = Boolean(config.singleton);
  const records = config.singleton ? [draftData[activeEditorSection]] : draftData[activeEditorSection];
  document.querySelector('#formList').innerHTML = records.map((record, index) => renderFormCard(config, record, index)).join('') || `
    <div class="editor-empty"><b>这里还没有内容</b><p>点击右上角“新增一项”开始维护。</p></div>`;
}

function renderFormCard(config, record, index) {
  if (config === editorSections.currentWork && !Array.isArray(record.stages)) record.stages = defaultStages();
  const controls = config.singleton ? '' : `<div class="item-controls"><button type="button" class="mini-button" data-move="up" data-index="${index}" aria-label="上移">↑</button><button type="button" class="mini-button" data-move="down" data-index="${index}" aria-label="下移">↓</button><button type="button" class="mini-button delete" data-delete-index="${index}">删除</button></div>`;
  const title = config.singleton ? '看板基本信息' : `${config.label} ${String(index + 1).padStart(2, '0')}`;
  return `<article class="form-card" data-record-index="${index}"><div class="form-card-head"><b>${title}</b>${controls}</div><div class="form-grid">${config.fields.map(field => renderField(field, record[field[0]], index)).join('')}</div></article>`;
}

function renderField([name, label, type, options], value, index) {
  if (type === 'stages') {
    return `<div class="form-field wide"><span>${escapeHtml(label)}</span><div class="managed-stages"><div class="stage-edit-list">${(value || []).map((stage, stageIndex) => renderManagedStage(stage, index, stageIndex)).join('')}</div><button type="button" class="quiet" data-add-managed-stage="${index}">＋ 添加阶段</button></div></div>`;
  }
  if (type === 'operatorChecklist') {
    return `<div class="form-field wide"><span>${escapeHtml(label)}</span><div class="managed-operators"><div class="operator-edit-list">${(value || []).map((operator, operatorIndex) => renderManagedOperator(operator, index, operatorIndex)).join('')}</div><button type="button" class="quiet" data-add-managed-operator="${index}">＋ 添加算子</button></div></div>`;
  }
  const wide = type === 'textarea' || type === 'list' || type === 'stages' || type === 'operatorChecklist' ? ' wide' : '';
  const displayValue = type === 'list' ? (value || []).join('，') : (value ?? '');
  let control;
  if (type === 'textarea') {
    control = `<textarea data-field="${name}" data-index="${index}" rows="3">${escapeHtml(displayValue)}</textarea>`;
  } else if (type === 'select') {
    control = `<select data-field="${name}" data-index="${index}">${options.map(option => `<option value="${escapeHtml(option)}" ${option === value ? 'selected' : ''}>${escapeHtml(statusLabels[option] || option)}</option>`).join('')}</select>`;
  } else {
    control = `<input type="${type === 'list' ? 'text' : type}" ${type === 'number' ? 'min="0" max="100"' : ''} value="${escapeHtml(displayValue)}" data-field="${name}" data-index="${index}" />`;
  }
  return `<label class="form-field${wide}"><span>${escapeHtml(label)}</span>${control}</label>`;
}

function renderManagedStage(stage, recordIndex, stageIndex) {
  return `<div class="stage-edit-row" data-managed-record-index="${recordIndex}" data-managed-stage-index="${stageIndex}"><input value="${escapeHtml(stage.name)}" data-managed-stage-field="name" aria-label="阶段名称" /><select data-managed-stage-field="status" aria-label="阶段状态">${renderStageStatusOptions(stage.status)}</select><button type="button" class="mini-button delete" data-delete-managed-stage>删除</button></div>`;
}

function renderManagedOperator(operator, recordIndex, operatorIndex) {
  return `<div class="operator-edit-row" data-managed-plan-index="${recordIndex}" data-managed-operator-index="${operatorIndex}"><label><input type="checkbox" data-managed-operator-field="done" ${operator.done ? 'checked' : ''}><span>完成</span></label><input value="${escapeHtml(operator.name)}" data-managed-operator-field="name" aria-label="算子名称" placeholder="输入算子名称" /><button type="button" class="mini-button delete" data-delete-managed-operator>删除</button></div>`;
}

function updateDraftFromControl(control) {
  const config = editorSections[activeEditorSection];
  const field = config.fields.find(item => item[0] === control.dataset.field);
  if (!field) return;
  let value = control.value;
  if (field[2] === 'number') value = Math.max(0, Math.min(100, Number(value) || 0));
  if (field[2] === 'list') value = value.split(/[,，\n]/).map(item => item.trim()).filter(Boolean);
  if (config.singleton) draftData[activeEditorSection][field[0]] = value;
  else draftData[activeEditorSection][Number(control.dataset.index)][field[0]] = value;
}

function updateManagedStageControl(control) {
  const row = control.closest('[data-managed-record-index]');
  if (!row) return;
  const record = draftData.currentWork[Number(row.dataset.managedRecordIndex)];
  record.stages[Number(row.dataset.managedStageIndex)][control.dataset.managedStageField] = control.value;
}

function updateManagedOperatorControl(control) {
  const row = control.closest('[data-managed-plan-index]');
  if (!row) return;
  const operator = draftData.plans[Number(row.dataset.managedPlanIndex)].operators[Number(row.dataset.managedOperatorIndex)];
  operator[control.dataset.managedOperatorField] = control.type === 'checkbox' ? control.checked : control.value;
}

function addItem() {
  const config = editorSections[activeEditorSection];
  if (!config.create) return;
  draftData[activeEditorSection].push(config.create());
  renderEditor();
  document.querySelector('#formList .form-card:last-child')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function deleteItem(index) {
  if (!confirm('确认删除这一项？保存全部更改后才会生效。')) return;
  draftData[activeEditorSection].splice(index, 1);
  renderEditor();
}

function moveItem(index, direction) {
  const items = draftData[activeEditorSection];
  const target = direction === 'up' ? index - 1 : index + 1;
  if (target < 0 || target >= items.length) return;
  [items[index], items[target]] = [items[target], items[index]];
  renderEditor();
}

function saveEditor() {
  validateData(draftData);
  boardData = clone(draftData);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(boardData));
  render();
  const message = document.querySelector('#editorMessage');
  message.textContent = '已保存到当前浏览器。需要同步到 GitHub 时，请导出数据并提交 plan_data.json。';
  message.classList.add('success');
}

function downloadData() {
  const data = draftData || boardData;
  const blob = new Blob([JSON.stringify(data, null, 2) + '\n'], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url; link.download = 'plan_data.json'; link.click();
  URL.revokeObjectURL(url);
}

document.querySelector('#editButton').addEventListener('click', () => openEditor('meta'));
document.querySelector('#exportButton').addEventListener('click', downloadData);
document.querySelector('#downloadButton').addEventListener('click', downloadData);
document.querySelector('#applyButton').addEventListener('click', saveEditor);
document.querySelector('#cancelEditButton').addEventListener('click', () => document.querySelector('#editorDialog').close());
document.querySelector('#closeEditorButton').addEventListener('click', () => document.querySelector('#editorDialog').close());
document.querySelector('#addItemButton').addEventListener('click', addItem);
document.querySelector('#quickAddTaskButton').addEventListener('click', addWorkCard);
document.querySelectorAll('[data-editor-section]').forEach(button => button.addEventListener('click', () => openEditor(button.dataset.editorSection)));
document.querySelector('main').addEventListener('click', event => {
  if (event.target.closest('[data-add-work-card]')) return addWorkCard();
  const saveTask = event.target.closest('[data-save-task]');
  if (saveTask) return saveWorkCard(Number(saveTask.dataset.saveTask));
  const deleteTask = event.target.closest('[data-delete-task]');
  if (deleteTask) return deleteWorkCard(Number(deleteTask.dataset.deleteTask));
  const card = event.target.closest('[data-task-card]');
  if (card && event.target.closest('[data-card-add-stage]')) {
    const index = Number(card.dataset.taskCard);
    boardData.currentWork[index].stages.push({ name: '新阶段', status: 'planned' });
    dirtyTasks.add(index);
    return renderCurrentWork();
  }
  if (card && event.target.closest('[data-card-delete-stage]')) {
    const index = Number(card.dataset.taskCard);
    const stageIndex = Number(event.target.closest('[data-stage-index]').dataset.stageIndex);
    boardData.currentWork[index].stages.splice(stageIndex, 1);
    dirtyTasks.add(index);
    return renderCurrentWork();
  }
  const inlineEdit = event.target.closest('[data-inline-edit-task]');
  if (inlineEdit) return startInlineTaskEdit(Number(inlineEdit.dataset.inlineEditTask));
  if (event.target.closest('[data-inline-cancel]')) return cancelInlineTaskEdit();
  if (event.target.closest('[data-inline-save]')) return saveInlineTaskEdit();
  if (event.target.closest('[data-inline-add-stage]')) {
    inlineTaskDraft.stages.push({ name: '自定义阶段', status: 'planned' });
    return renderCurrentWork();
  }
  const deleteStage = event.target.closest('[data-inline-delete-stage]');
  if (deleteStage) {
    inlineTaskDraft.stages.splice(Number(deleteStage.dataset.inlineDeleteStage), 1);
    return renderCurrentWork();
  }
  const button = event.target.closest('[data-edit-record]');
  if (!button) return;
  openEditor(button.dataset.editRecord, Number(button.dataset.index));
});
document.querySelector('main').addEventListener('dragstart', event => {
  const handle = event.target.closest('[data-drag-task]');
  if (!handle) return;
  draggedTaskIndex = Number(handle.dataset.dragTask);
  event.dataTransfer.effectAllowed = 'move';
  event.dataTransfer.setData('text/plain', String(draggedTaskIndex));
  requestAnimationFrame(() => handle.closest('[data-task-card]')?.classList.add('dragging'));
});
document.querySelector('main').addEventListener('dragover', event => {
  if (draggedTaskIndex === null) return;
  const target = event.target.closest('[data-task-card]');
  if (!target || Number(target.dataset.taskCard) === draggedTaskIndex) return;
  event.preventDefault();
  event.dataTransfer.dropEffect = 'move';
  document.querySelectorAll('.work-card.drag-over').forEach(card => card.classList.remove('drag-over'));
  target.classList.add('drag-over');
});
document.querySelector('main').addEventListener('drop', event => {
  const target = event.target.closest('[data-task-card]');
  if (draggedTaskIndex === null || !target) return;
  event.preventDefault();
  const targetIndex = Number(target.dataset.taskCard);
  reorderWorkCard(draggedTaskIndex, targetIndex);
  draggedTaskIndex = null;
});
document.querySelector('main').addEventListener('dragend', () => {
  draggedTaskIndex = null;
  document.querySelectorAll('.work-card.dragging, .work-card.drag-over').forEach(card => card.classList.remove('dragging', 'drag-over'));
});
document.querySelector('main').addEventListener('input', event => {
  const card = event.target.closest('[data-task-card]');
  if (card && event.target.matches('[data-card-field]')) {
    const index = Number(card.dataset.taskCard);
    boardData.currentWork[index][event.target.dataset.cardField] = event.target.textContent.trim();
    return markTaskDirty(index, card);
  }
  if (card && event.target.matches('[data-stage-name]')) {
    const index = Number(card.dataset.taskCard);
    const stageIndex = Number(event.target.closest('[data-stage-index]').dataset.stageIndex);
    boardData.currentWork[index].stages[stageIndex].name = event.target.textContent.trim();
    return markTaskDirty(index, card);
  }
  if (event.target.matches('[data-inline-field]')) updateInlineTaskControl(event.target);
  if (event.target.matches('[data-inline-stage-field]')) updateInlineStageControl(event.target);
});
document.querySelector('main').addEventListener('change', event => {
  if (event.target.matches('[data-plan-operator-toggle]')) {
    const plan = boardData.plans[Number(event.target.dataset.planIndex)];
    plan.operators[Number(event.target.dataset.operatorIndex)].done = event.target.checked;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(boardData));
    return renderPlans();
  }
  const card = event.target.closest('[data-task-card]');
  if (card && event.target.matches('[data-card-select]')) {
    const index = Number(card.dataset.taskCard);
    boardData.currentWork[index][event.target.dataset.cardSelect] = event.target.value;
    return markTaskDirty(index, card);
  }
  if (card && event.target.matches('[data-card-stage-status]')) {
    const index = Number(card.dataset.taskCard);
    const stageIndex = Number(event.target.closest('[data-stage-index]').dataset.stageIndex);
    boardData.currentWork[index].stages[stageIndex].status = event.target.value;
    event.target.className = `stage-status-select ${event.target.value}`;
    event.target.closest('[data-stage-index]').className = `stage-item stage-${event.target.value}`;
    return markTaskDirty(index, card);
  }
  if (event.target.matches('[data-direct-stage-status]')) {
    const task = boardData.currentWork[Number(event.target.dataset.taskIndex)];
    task.stages[Number(event.target.dataset.stageIndex)].status = event.target.value;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(boardData));
    return renderCurrentWork();
  }
  if (event.target.matches('[data-inline-field]')) updateInlineTaskControl(event.target);
  if (event.target.matches('[data-inline-stage-field]')) updateInlineStageControl(event.target);
});
document.querySelector('main').addEventListener('focusout', event => {
  if (!event.target.matches('[data-direct-issue-summary]')) return;
  const index = Number(event.target.dataset.directIssueSummary);
  boardData.currentWork[index].currentIssueSummary = event.target.textContent.trim();
  localStorage.setItem(STORAGE_KEY, JSON.stringify(boardData));
});
document.querySelector('main').addEventListener('keydown', event => {
  if (event.target.matches('.work-id, .area-text, .work-card h3') && event.key === 'Enter') {
    event.preventDefault();
    return event.target.blur();
  }
  if (event.target.closest('[data-task-card]') && (event.metaKey || event.ctrlKey) && event.key === 'Enter') {
    event.preventDefault();
    return saveWorkCard(Number(event.target.closest('[data-task-card]').dataset.taskCard));
  }
  if (event.target.matches('[data-direct-issue-summary]') && (event.metaKey || event.ctrlKey) && event.key === 'Enter') event.target.blur();
});

document.querySelector('#editorNav').addEventListener('click', event => {
  const button = event.target.closest('[data-editor-nav]');
  if (!button) return;
  activeEditorSection = button.dataset.editorNav;
  renderEditor();
});

document.querySelector('#formList').addEventListener('input', event => {
  if (event.target.matches('[data-field]')) updateDraftFromControl(event.target);
  if (event.target.matches('[data-managed-stage-field]')) updateManagedStageControl(event.target);
  if (event.target.matches('[data-managed-operator-field]')) updateManagedOperatorControl(event.target);
});
document.querySelector('#formList').addEventListener('change', event => {
  if (event.target.matches('[data-field]')) updateDraftFromControl(event.target);
  if (event.target.matches('[data-managed-stage-field]')) updateManagedStageControl(event.target);
  if (event.target.matches('[data-managed-operator-field]')) updateManagedOperatorControl(event.target);
});
document.querySelector('#formList').addEventListener('click', event => {
  const addOperator = event.target.closest('[data-add-managed-operator]');
  if (addOperator) {
    const plan = draftData.plans[Number(addOperator.dataset.addManagedOperator)];
    if (!Array.isArray(plan.operators)) plan.operators = [];
    plan.operators.push({ name: '新算子', done: false });
    return renderEditor();
  }
  const deleteOperator = event.target.closest('[data-delete-managed-operator]');
  if (deleteOperator) {
    const row = deleteOperator.closest('[data-managed-plan-index]');
    draftData.plans[Number(row.dataset.managedPlanIndex)].operators.splice(Number(row.dataset.managedOperatorIndex), 1);
    return renderEditor();
  }
  const addStage = event.target.closest('[data-add-managed-stage]');
  if (addStage) {
    draftData.currentWork[Number(addStage.dataset.addManagedStage)].stages.push({ name: '自定义阶段', status: 'planned' });
    return renderEditor();
  }
  const deleteStage = event.target.closest('[data-delete-managed-stage]');
  if (deleteStage) {
    const row = deleteStage.closest('[data-managed-record-index]');
    draftData.currentWork[Number(row.dataset.managedRecordIndex)].stages.splice(Number(row.dataset.managedStageIndex), 1);
    return renderEditor();
  }
  const remove = event.target.closest('[data-delete-index]');
  if (remove) return deleteItem(Number(remove.dataset.deleteIndex));
  const move = event.target.closest('[data-move]');
  if (move) moveItem(Number(move.dataset.index), move.dataset.move);
});

document.querySelector('#resetButton').addEventListener('click', () => {
  if (!confirm('确认放弃当前编辑并恢复仓库中的默认数据？')) return;
  draftData = clone(repositoryData);
  localStorage.removeItem(STORAGE_KEY);
  localStorage.removeItem(LEGACY_STORAGE_KEY);
  renderEditor();
  document.querySelector('#editorMessage').textContent = '已载入仓库版本，点击“保存全部更改”后生效。';
});

document.querySelector('#importInput').addEventListener('change', async event => {
  const file = event.target.files[0];
  if (!file) return;
  try {
    const data = JSON.parse(await file.text());
    validateData(data);
    draftData = data;
    renderEditor();
    document.querySelector('#editorMessage').textContent = '数据已导入，检查后点击“保存全部更改”。';
  } catch (error) {
    document.querySelector('#editorMessage').textContent = `导入失败：${error.message}`;
  }
  event.target.value = '';
});

document.querySelector('#statusFilters').addEventListener('click', event => {
  const button = event.target.closest('button[data-status]');
  if (!button) return;
  activeStatus = button.dataset.status;
  document.querySelectorAll('#statusFilters button').forEach(item => item.classList.toggle('active', item === button));
  renderCurrentWork();
});

document.querySelector('#editorDialog').addEventListener('close', () => {
  draftData = null;
  document.querySelector('#editorMessage').classList.remove('success');
});

loadData().catch(error => {
  document.querySelector('main').innerHTML = `<section class="load-error"><h1>看板加载失败</h1><p>${escapeHtml(error.message)}</p><p>请通过本地 HTTP 服务或 GitHub Pages 打开页面，不要直接双击 index.html。</p></section>`;
});
