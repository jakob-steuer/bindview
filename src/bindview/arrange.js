// The page in which a figure is arranged by hand (bindview/arrange.py). It knows the figure by what bindview writes into it:
// groups of residues and contacts with data-residue, and the graph of the figure, whose nodes have the places.
(function () {
  "use strict";
  var given = JSON.parse(document.getElementById("graph").textContent), exact = window.EXACT || null;
  var orig = given.graph, bonds = given.bonds, now = copy(orig), svg = null, items = {}, byId = {}, STEP = 15;
  var picked = null, drag = null, past = [], busy = false, again = false;
  function copy(x) { return JSON.parse(JSON.stringify(x)); }
  function $(id) { return document.getElementById(id); }
  function say(words) { $("said").textContent = words; }

  // ------------------------------------------------------------ reading a figure
  function read() {
    svg = document.querySelector("#figure svg"); items = {}; byId = {};
    now.nodes.forEach(function (n) { byId[n.id] = n; });
    svg.querySelectorAll("g.residue[data-residue]").forEach(function (g) {
      var id = "R:" + g.getAttribute("data-residue"), n = byId[id];
      if (!n || n.x === null || n.x === undefined) { return; }
      var it = items[id] = items[id] || { id: id, n: n, parts: [], links: [], spokes: [], lining: [], dx: 0, dy: 0, turn: 0, flip: false };
      it.parts = it.parts.concat(leaves(g)); it.group = g;
      it.targets = now.contacts.filter(function (c) { return c.residue === id && c.ligand; })
                               .map(function (c) { return byId[c.ligand]; }).filter(Boolean)
                               .map(function (t) { return [t.x, t.y]; });
      g.style.cursor = "grab";
      g.addEventListener("pointerdown", function (e) { begin(e, id); });
      g.addEventListener("dblclick", function () { back(id); });
    });
    svg.querySelectorAll("g.interaction[data-residue]").forEach(function (g) {
      var it = items["R:" + g.getAttribute("data-residue")]; if (it) { it.links.push(link(g, it)); }
    });
    svg.querySelectorAll("g.contact[data-residue]").forEach(function (g) {
      var it = items["R:" + g.getAttribute("data-residue")]; if (it) { it.spokes.push(g); }
    });
    svg.querySelectorAll("g.lining[data-residues]").forEach(function (g) {
      g.getAttribute("data-residues").split(" ").forEach(function (r) { if (items["R:" + r]) { items["R:" + r].lining.push(g); } });
    });
    Object.keys(items).forEach(function (id) { shape(items[id]); });
    var key = svg.querySelector("g.legend");
    if (key && byId.legend) {
      items.legend = { id: "legend", n: byId.legend, key: key, home: key.getAttribute("transform") || "", dx: 0, dy: 0, turn: 0,
                       flip: false, parts: [], links: [], spokes: [], lining: [] };
      key.style.cursor = "grab";
      key.addEventListener("pointerdown", function (e) { begin(e, "legend"); });
    }
    svg.addEventListener("pointermove", function (e) {
      if (!drag) { return; }
      var p = place(e), it = items[drag.id]; it.dx = p.x - drag.x; it.dy = p.y - drag.y; show(it);
    });
    svg.addEventListener("pointerup", function () { if (drag) { drag = null; done(); } });
    svg.addEventListener("pointerdown", function (e) { if (!e.target.closest("g.residue, g.legend")) { pick(null); } });
    svg.addEventListener("wheel", function (e) {
      if (!picked || !items[picked] || items[picked].key) { return; }
      e.preventDefault(); turn((e.deltaY > 0 ? 1 : -1) * 5); later();
    }, { passive: false });
    if (picked && items[picked]) { pick(picked); }
    write();
  }
  function leaves(g) {
    var out = [];
    (function walk(el) {
      Array.prototype.forEach.call(el.children, function (c) {
        if (c.tagName === "g") { walk(c); } else if (c.tagName !== "title") {
          out.push({ el: c, text: c.tagName === "text", x: +c.getAttribute("x"), y: +c.getAttribute("y"),
                     size: +(c.getAttribute("font-size") || 10) });
        }
      });
    })(g);
    return out;
  }
  function ends(g) {
    var pts = [];
    g.querySelectorAll("line").forEach(function (l) {
      pts.push([+l.getAttribute("x1"), +l.getAttribute("y1")], [+l.getAttribute("x2"), +l.getAttribute("y2")]);
    });
    return pts;
  }
  function link(g, it) {                             // a contact's line: its end at the residue, its end at the ligand
    var pts = ends(g), best = [pts[0] || [0, 0], pts[1] || pts[0] || [0, 0]], far = -1, n = it.n;
    pts.forEach(function (p) { pts.forEach(function (q) {
      var d = Math.hypot(p[0] - q[0], p[1] - q[1]); if (d > far) { far = d; best = [p, q]; }
    }); });
    var near = Math.hypot(best[0][0] - n.x, best[0][1] - n.y) <= Math.hypot(best[1][0] - n.x, best[1][1] - n.y) ? 0 : 1;
    return { a: best[near], b: best[1 - near], all: leaves(g) };
  }
  function shape(it) {                               // what faces the ligand by itself: an eyelash, the tip of a bubble
    var n = it.n;
    if (n.form === "eyelash") {
      it.arcs = it.parts.filter(function (l) { return l.el.tagName === "polyline"; });
      it.lashes = it.parts.filter(function (l) { return l.el.tagName === "line"; });
      if (!it.arcs.length) { it.arcs = null; return; }
      var p = it.arcs[0].el.getAttribute("points").trim().split(/\s+/)[0].split(",");
      it.radius = Math.hypot(+p[0] - n.x, +p[1] - n.y);
      var l = it.lashes[0] && it.lashes[0].el;
      it.lash = l ? Math.hypot(+l.getAttribute("x2") - n.x, +l.getAttribute("y2") - n.y) - it.radius : 5;
    } else if (n.form === "bubble") {
      it.drop = it.parts.filter(function (l) { return l.el.tagName === "path" && / A /.test(" " + l.el.getAttribute("d").replace(/A/g, " A ")); });
      it.drop.forEach(function (l) {
        var m = l.el.getAttribute("d").match(/A\s*([\d.]+)/); l.radius = m ? +m[1] : 15;
      });
    }
  }

  // ------------------------------------------------------------ drawing the preview
  function moved(it, p) {
    var n = it.n, x = p[0] - n.x, y = p[1] - n.y, a0 = n.angle || 0, c, s, u, v;
    if (it.flip) {
      c = Math.cos(-a0); s = Math.sin(-a0); u = x * c - y * s; v = -(x * s + y * c);
      c = Math.cos(a0); s = Math.sin(a0); x = u * c - v * s; y = u * s + v * c;
    }
    c = Math.cos(it.turn); s = Math.sin(it.turn);
    return [x * c - y * s + n.x + it.dx, x * s + y * c + n.y + it.dy];
  }
  function matrix(it) {
    var o = moved(it, [0, 0]), e = moved(it, [1, 0]), f = moved(it, [0, 1]);
    return "matrix(" + [e[0] - o[0], e[1] - o[1], f[0] - o[0], f[1] - o[1], o[0], o[1]].join(" ") + ")";
  }
  function clusters(c, targets) {
    var full = 2 * Math.PI, deg = Math.PI / 180;
    var ang = targets.map(function (t) { return ((Math.atan2(t[1] - c[1], t[0] - c[0]) % full) + full) % full; })
                     .sort(function (a, b) { return a - b; });
    if (!ang.length) { return []; }
    if (ang.length === 1) { return [[ang[0], 56 * deg]]; }
    var big = -1, at = 0;
    ang.forEach(function (a, i) { var g = ((ang[(i + 1) % ang.length] - a) % full + full) % full; if (g >= big) { big = g; at = i; } });
    var spread = full - big, start = ang[(at + 1) % ang.length];
    if (spread <= 110 * deg) { return [[start + spread / 2, 56 * deg]]; }
    var run = ang.map(function (a) { return ((a - start) % full + full) % full; }).sort(function (a, b) { return a - b; });
    var cut = 0, gap = -1;
    for (var i = 0; i + 1 < run.length; i++) { if (run[i + 1] - run[i] >= gap) { gap = run[i + 1] - run[i]; cut = i; } }
    return [run.slice(0, cut + 1), run.slice(cut + 1)].map(function (p) { return [start + (p[0] + p[p.length - 1]) / 2, 40 * deg]; });
  }
  function eyelash(it) {
    var c = [it.n.x + it.dx, it.n.y + it.dy], parent = it.arcs[0].el.parentNode, like = it.arcs[0].el, lash = it.lashes[0].el;
    it.arcs.concat(it.lashes).forEach(function (l) { l.el.remove(); });
    it.arcs = []; it.lashes = [];
    clusters(c, it.targets).forEach(function (k) {
      var pts = [], i, a, n = k[1] > 50 * Math.PI / 180 ? 7 : 5;
      for (i = 0; i <= 24; i++) { a = k[0] - k[1] + 2 * k[1] * i / 24; pts.push((c[0] + it.radius * Math.cos(a)).toFixed(2) + "," + (c[1] + it.radius * Math.sin(a)).toFixed(2)); }
      var arc = like.cloneNode(false); arc.setAttribute("points", pts.join(" ")); arc.removeAttribute("transform");
      parent.insertBefore(arc, parent.firstChild); it.arcs.push({ el: arc });
      for (i = 0; i < n; i++) {
        a = k[0] - k[1] + 2 * k[1] * (i + 0.5) / n;
        var l = lash.cloneNode(false); l.removeAttribute("transform");
        l.setAttribute("x1", c[0] + it.radius * Math.cos(a)); l.setAttribute("y1", c[1] + it.radius * Math.sin(a));
        l.setAttribute("x2", c[0] + (it.radius + it.lash) * Math.cos(a)); l.setAttribute("y2", c[1] + (it.radius + it.lash) * Math.sin(a));
        parent.insertBefore(l, parent.firstChild); it.lashes.push({ el: l });
      }
    });
  }
  function drop(it, l) {
    var x = it.n.x + it.dx, y = it.n.y + it.dy, r = l.radius, tip = 1.3, phi = Math.acos(1 / tip), tx = 0, ty = 0;
    it.targets.forEach(function (t) { tx += t[0] / it.targets.length; ty += t[1] / it.targets.length; });
    var th = Math.atan2(ty - y, tx - x), f = function (k, a) { return (x + k * r * Math.cos(a)).toFixed(2) + " " + (y + k * r * Math.sin(a)).toFixed(2); };
    l.el.setAttribute("d", "M " + f(tip, th) + " L " + f(1, th + phi) + " A " + r + " " + r + " 0 1 1 " + f(1, th - phi) + " Z");
  }
  // The spokes on the atoms of the ligand: fanned towards the residue, between the atom's bonds.
  function spokes(it) {
    it.spokes.forEach(function (g) {
      var lines = Array.prototype.slice.call(g.querySelectorAll("line")); if (!lines.length) { return; }
      var atoms = (g.getAttribute("data-ligand-atoms") || "").split(",").map(function (a) { return byId["L:" + a]; }).filter(Boolean);
      atoms.forEach(function (n) {
        var mine = lines.filter(function (l) { return Math.hypot(+l.getAttribute("x1") - n.x, +l.getAttribute("y1") - n.y) < 26; });
        if (!mine.length) { return; }
        var r0 = Math.hypot(+mine[0].getAttribute("x1") - n.x, +mine[0].getAttribute("y1") - n.y);
        var r1 = Math.hypot(+mine[0].getAttribute("x2") - n.x, +mine[0].getAttribute("y2") - n.y);
        var want = Math.atan2(it.n.y + it.dy - n.y, it.n.x + it.dx - n.x), off = 21 * Math.PI / 180;
        var dirs = (bonds[n.id] || []).map(function (id) { return byId[id]; }).filter(Boolean)
                   .map(function (q) { return Math.atan2(q.y - n.y, q.x - n.x); });
        var free = function (a) { return dirs.every(function (b) { return Math.abs(Math.atan2(Math.sin(a - b), Math.cos(a - b))) >= off; }); };
        var mid = null;
        for (var k = 0; k <= 60 && mid === null; k++) {
          [1, -1].forEach(function (sg) { var a = want + sg * 3 * k * Math.PI / 180; if (mid === null && free(a)) { mid = a; } });
        }
        if (mid === null) { return; }
        var like = mine[0]; mine.forEach(function (l) { if (l !== like) { l.remove(); } });
        [-0.42, 0, 0.42].filter(function (d) { return free(mid + d); }).forEach(function (d, i) {
          var l = i ? like.cloneNode(false) : like, a = mid + d;
          l.setAttribute("x1", n.x + r0 * Math.cos(a)); l.setAttribute("y1", n.y + r0 * Math.sin(a));
          l.setAttribute("x2", n.x + r1 * Math.cos(a)); l.setAttribute("y2", n.y + r1 * Math.sin(a));
          if (i) { g.appendChild(l); }
        });
      });
    });
  }
  function shift(l, q, mid) { l.el.setAttribute("transform", "translate(" + (q[0] - mid[0]) + " " + (q[1] - mid[1]) + ")"); }
  function show(it) {
    if (it.key) { it.key.setAttribute("transform", "translate(" + it.dx + " " + it.dy + ") " + it.home); write(); return; }
    var m = matrix(it);
    it.parts.forEach(function (l) {
      if (!l.el.isConnected) { return; }
      var mid = [l.x, l.y - 0.36 * l.size];
      if (l.text) { shift(l, moved(it, mid), mid); }
      else if (it.drop && it.drop.indexOf(l) >= 0) { drop(it, l); }
      else if (!it.arcs) { l.el.setAttribute("transform", m); }
    });
    if (it.arcs) { eyelash(it); }
    spokes(it);
    // The wall of the pocket is bindview's to draw: where the residue has gone, what is on the page is of its old place.
    it.lining.forEach(function (g) { g.setAttribute("opacity", "0.25"); });
    it.links.forEach(function (k) {
      // The end at the ligand stays, the end at the residue goes with it: a turn and a stretch about the former.
      var q = moved(it, k.a), ax = k.a[0] - k.b[0], ay = k.a[1] - k.b[1], bx = q[0] - k.b[0], by = q[1] - k.b[1];
      var d = ax * ax + ay * ay || 1, c = (ax * bx + ay * by) / d, s = (ax * by - ay * bx) / d;
      var mk = "matrix(" + [c, s, -s, c, k.b[0] - c * k.b[0] + s * k.b[1], k.b[1] - s * k.b[0] - c * k.b[1]].join(" ") + ")";
      k.all.forEach(function (l) {
        var mid = [l.x, l.y - 0.36 * l.size], x = mid[0] - k.b[0], y = mid[1] - k.b[1];
        if (l.text) { shift(l, [c * x - s * y + k.b[0], s * x + c * y + k.b[1]], mid); }
        else { l.el.setAttribute("transform", mk); }
      });
    });
    write();
  }

  // ------------------------------------------------------------ what the hand does
  function place(e) {
    var p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY;
    return p.matrixTransform(svg.getScreenCTM().inverse());
  }
  function begin(e, id) {
    e.preventDefault(); pick(id);
    var p = place(e), it = items[id];
    drag = { id: id, x: p.x - it.dx, y: p.y - it.dy };
    try { svg.setPointerCapture(e.pointerId); } catch (err) { /* a made-up event has no pointer */ }
  }
  function pick(id) {
    picked = id;
    svg.querySelectorAll("g.residue, g.legend").forEach(function (g) {
      var r = g.getAttribute("data-residue");
      g.classList.toggle("picked", !!id && (r ? "R:" + r === id : id === "legend"));
    });
    $("which").textContent = id ? (byId[id].label || id) : "nothing";
  }
  function turnable(it) { return it.n.form === "fragment"; }
  function turn(by) {
    var it = items[picked]; if (!it || it.key) { return; }
    if (!turnable(it)) { say("It is drawn without atoms and faces the ligand by itself: it is moved, not turned."); return; }
    it.turn += by * Math.PI / 180; show(it);
  }
  function flip() {
    var it = items[picked]; if (!it || it.key) { return; }
    if (!turnable(it)) { say("It is drawn without atoms: there is nothing to flip."); return; }
    it.flip = !it.flip; show(it); done();
  }
  function back(id) {
    var o = orig.nodes.filter(function (n) { return n.id === id; })[0], n = byId[id];
    now.nodes[now.nodes.indexOf(n)] = copy(o); past.push(null); redraw(now, true);
  }
  document.addEventListener("keydown", function (e) {
    if (e.target.tagName === "TEXTAREA") { return; }
    if ((e.metaKey || e.ctrlKey) && e.key === "z") { e.preventDefault(); undo(); return; }
    if (!picked || !items[picked]) { return; }
    var it = items[picked], step = e.shiftKey ? 10 : 2, k = e.key;
    if (k === "r" || k === "R") { turn(k === "R" ? -STEP : STEP); later(); }
    else if (k === "f") { flip(); }
    else if (k === "Escape") { pick(null); }
    else if (k.indexOf("Arrow") === 0) {
      it.dx += k === "ArrowLeft" ? -step : k === "ArrowRight" ? step : 0;
      it.dy += k === "ArrowUp" ? -step : k === "ArrowDown" ? step : 0; show(it); later();
    } else { return; }
    e.preventDefault();
  });
  var timer = null;
  function later() { clearTimeout(timer); timer = setTimeout(done, 350); }
  function touched() { return Object.keys(items).some(function (id) { var it = items[id]; return it.dx || it.dy || it.turn || it.flip; }); }
  function edited() {
    var g = copy(now);
    g.nodes.forEach(function (n) {
      var it = items[n.id]; if (!it) { return; }
      if (it.key) { n.offset = [(n.offset || [0, 0])[0] + it.dx, (n.offset || [0, 0])[1] + it.dy]; }
      n.x += it.dx; n.y += it.dy;
      if (it.flip) { if (n.mirror) { delete n.mirror; } else { n.mirror = true; } }
      if (it.turn) { n.angle = (n.angle || 0) + it.turn; }
    });
    return g;
  }
  // Let go: bindview draws the figure from the layout (where it is behind the page), and that is the new figure.
  function done() {
    if (drag || !touched()) { return; }
    past.push(copy(now)); redraw(edited(), false);
  }
  function undo() {
    var g = past.pop(); while (g === null) { g = past.pop(); }
    if (g) { redraw(g, true); }
  }
  function redraw(graph, whole) {
    if (!exact) {
      if (whole) { now = copy(orig); $("figure").innerHTML = given.svg; now = copy(orig); read(); replay(graph); } else { write(); }
      return;
    }
    if (busy) { again = true; return; }
    busy = true; say("drawing …");
    fetch(exact + "render", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(graph) })
      .then(function (r) { return r.json(); }).then(function (a) {
        busy = false;
        if (a.error) { say(a.error); return; }
        $("figure").innerHTML = a.svg; now = a.graph; read(); say(a.said || "");
        if (again) { again = false; done(); }
      }).catch(function (err) { busy = false; say("bindview does not answer: " + err); });
  }
  function replay(graph) {                           // without bindview: the preview of a layout, from the first figure
    graph.nodes.forEach(function (n) {
      var it = items[n.id], o = byId[n.id]; if (!it || o.x === null) { return; }
      if (it.key) { it.dx = (n.offset || [0, 0])[0] - (o.offset || [0, 0])[0]; it.dy = (n.offset || [0, 0])[1] - (o.offset || [0, 0])[1]; }
      else { it.dx = n.x - o.x; it.dy = n.y - o.y; it.turn = (n.angle || 0) - (o.angle || 0); it.flip = !!n.mirror !== !!o.mirror; }
      if (it.dx || it.dy || it.turn || it.flip) { show(it); }
    });
  }

  // ------------------------------------------------------------ what comes out
  function num(v) { return (Math.round(v * 10) / 10).toString(); }
  function calls() {
    var g = edited(), was = {}, out = []; orig.nodes.forEach(function (n) { was[n.id] = n; });
    g.nodes.forEach(function (n) {
      var o = was[n.id]; if (!items[n.id] || !o) { return; }
      var name = JSON.stringify(n.id === "legend" ? "legend" : n.label), dx, dy;
      if (n.id === "legend") { dx = (n.offset || [0, 0])[0] - (o.offset || [0, 0])[0]; dy = (n.offset || [0, 0])[1] - (o.offset || [0, 0])[1]; }
      else { dx = n.x - o.x; dy = n.y - o.y; }
      if (!!n.mirror !== !!o.mirror) { out.push("graph.flip(" + name + ")"); }
      var d = Math.round(((n.angle || 0) - (o.angle || 0)) * 180 / Math.PI) % 360;
      if (d) { out.push("graph.rotate(" + name + ", " + d + ")"); }
      if (Math.abs(dx) > 0.05 || Math.abs(dy) > 0.05) { out.push("graph.move(" + name + ", " + num(dx) + ", " + num(dy) + ")"); }
    });
    return out;
  }
  function write() {
    var c = calls();
    $("calls").value = c.length ? ["graph = diagram.render().graph"].concat(c, ["figure = diagram.render(graph=graph)"]).join("\n")
                                : "# nothing is moved yet";
  }
  $("save").addEventListener("click", function () {
    var a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify(edited(), null, 1) + "\n"], { type: "application/json" }));
    a.download = "layout.json"; a.click(); URL.revokeObjectURL(a.href);
  });
  $("copy").addEventListener("click", function () { navigator.clipboard.writeText($("calls").value).then(function () { say("copied"); }); });
  $("reset").addEventListener("click", function () { past.push(copy(edited())); redraw(copy(orig), true); });
  $("undo").addEventListener("click", undo);
  $("turn").addEventListener("click", function () { turn(STEP); later(); });
  $("back").addEventListener("click", function () { turn(-STEP); later(); });
  $("flip").addEventListener("click", flip);
  if ($("done")) { $("done").addEventListener("click", function () {
    fetch(exact + "done", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(edited()) })
      .then(function () { say("The layout went back to Python. This page can be closed."); });
  }); }
  read();
  window.arrange = { items: function () { return items; }, edited: edited, calls: calls, show: show, pick: pick, turn: turn, flip: flip, done: done, undo: undo };
})();
