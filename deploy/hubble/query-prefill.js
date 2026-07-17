(function () {
  "use strict";

  var query = new URLSearchParams(window.location.search).get("gremlin");
  if (!query || !window.location.pathname.endsWith("/data-analyze")) {
    return;
  }

  var attempts = 0;
  var executed = false;
  var timer = window.setInterval(function () {
    attempts += 1;
    var editorElement = Array.prototype.find.call(
      document.querySelectorAll(".CodeMirror"),
      function (element) { return Boolean(element.CodeMirror); }
    );
    if (editorElement && editorElement.CodeMirror) {
      editorElement.CodeMirror.setValue(query);
      editorElement.CodeMirror.focus();

      var executeButton = document.querySelector('button[title="执行查询"]');
      if (executeButton && !executeButton.disabled && !executed) {
        executed = true;
        window.clearInterval(timer);
        window.setTimeout(function () {
          executeButton.click();
        }, 100);
        return;
      }
    }
    if (attempts >= 80) {
      window.clearInterval(timer);
    }
  }, 125);
}());
