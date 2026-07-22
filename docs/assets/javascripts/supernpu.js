(function () {
  function activateModel(root, target) {
    const buttons = root.querySelectorAll('[role="tab"]');
    const panels = root.querySelectorAll('[role="tabpanel"]');

    buttons.forEach((button) => {
      const selected = button.dataset.target === target;
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
    });

    panels.forEach((panel) => {
      panel.hidden = panel.id !== target;
    });
  }

  function hydrateModelSwitchers() {
    document.querySelectorAll("[data-model-switcher]").forEach((root) => {
      if (root.dataset.ready === "true") return;
      root.dataset.ready = "true";

      const buttons = Array.from(root.querySelectorAll('[role="tab"]'));
      buttons.forEach((button, index) => {
        button.addEventListener("click", () => activateModel(root, button.dataset.target));
        button.addEventListener("keydown", (event) => {
          if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
          event.preventDefault();
          let next;
          if (event.key === "Home") next = buttons[0];
          else if (event.key === "End") next = buttons[buttons.length - 1];
          else {
            const delta = event.key === "ArrowRight" ? 1 : -1;
            next = buttons[(index + delta + buttons.length) % buttons.length];
          }
          activateModel(root, next.dataset.target);
          next.focus();
        });
      });
    });
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(hydrateModelSwitchers);
  } else {
    document.addEventListener("DOMContentLoaded", hydrateModelSwitchers);
  }
})();
