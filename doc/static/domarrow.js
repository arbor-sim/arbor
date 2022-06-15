(()=>{
  function getNumberOrDef(val, def) {
    return typeof val === 'number' && !isNaN(val) ? val : def;
  }

  function isVisible(element) {
    return element && element.style.visibility !== 'hidden';
  }

  function inside(minX, minY, maxX, maxY, x1, y1) {
	  return minX <= x1 && x1 <= maxX && minY <= y1 && y1 <= maxY;
  }

  function intersectionPoint (x1, y1, x2, y2, minX, minY, maxX, maxY) {
    var min = Math.min, max = Math.max;
    var good = inside.bind(null, min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2));

    if ((x1 <= minX && x2 <= minX) || (y1 <= minY && y2 <= minY) || (x1 >= maxX && x2 >= maxX) || (y1 >= maxY && y2 >= maxY) || (inside(minX, minY, maxX, maxY, x1, y1) && inside(minX, minY, maxX, maxY, x2, y2)))
      return null;

    var m = (y2 - y1) / (x2 - x1);
    var y = m * (minX - x1) + y1;
    if (minY < y && y < maxY && good(minX, y)) return {x: minX, y:y};

    y = m * (maxX - x1) + y1;
    if (minY < y && y < maxY && good(maxX, y)) return {x: maxX, y:y};

    var x = (minY - y1) / m + x1;
    if (minX < x && x < maxX && good(x, minY)) return {x: x, y:minY};

    x = (maxY - y1) / m + x1;
    if (minX < x && x < maxX && good(x, maxY)) return {x: x, y:maxY};

    return null;
  }


  function adjustLine(from, to, line, trafo){
    var color = trafo && trafo.color || 'black';
    var W = trafo && trafo.width || 2;

    var fromB = parseFloat(from.style.top) ? null : from.getBoundingClientRect();
    var toB = parseFloat(to.style.top) ? null : to.getBoundingClientRect();
    var fromBStartY = (fromB ? window.scrollY + fromB.top : parseFloat(from.style.top));
    var fromBStartX = (fromB ? window.scrollX + fromB.left : parseFloat(from.style.left));
    var toBStartY = (toB ? window.scrollY + toB.top : parseFloat(to.style.top));
    var toBStartX = (toB ? window.scrollX + toB.left : parseFloat(to.style.left));
    var fromBWidth = (fromB ? fromB.width : parseFloat(from.style.width) || from.offsetWidth);
    var fromBHeight = (fromB ? fromB.height : parseFloat(from.style.height) || from.offsetHeight);
    var toBWidth = (toB ? toB.width : parseFloat(to.style.width) || to.offsetWidth);
    var toBHeight = (toB ? toB.height : parseFloat(to.style.height) || to.offsetHeight);

    var fT = fromBStartY + fromBHeight * getNumberOrDef(trafo && trafo.fromY, getNumberOrDef(trafo, 0.5));
    var tT = toBStartY + toBHeight * getNumberOrDef(trafo && trafo.toY, getNumberOrDef(trafo, 0.5));
    var fL = fromBStartX + fromBWidth * getNumberOrDef(trafo && trafo.fromX, getNumberOrDef(trafo, 0.5));
    var tL = toBStartX + toBWidth * getNumberOrDef(trafo && trafo.toX, getNumberOrDef(trafo, 0.5));

    var CA   = Math.abs(tT - fT);
    var CO   = Math.abs(tL - fL);
    var H    = Math.sqrt(CA*CA + CO*CO);
    var ANG  = 180 / Math.PI * Math.acos( CO/H );

    if((fT >= tT || fL >= tL) && (tT >= fT || tL >= fL)) {
      ANG *= -1;
    }

    if (trafo && trafo.onlyVisible) {
      var arrangeFrom = intersectionPoint(fL, fT, tL, tT, fromBStartX, fromBStartY, fromBStartX + fromBWidth, fromBStartY + fromBHeight);
      var arrangeTo = intersectionPoint(fL, fT, tL, tT, toBStartX, toBStartY, toBStartX + toBWidth, toBStartY + toBHeight);

      if (arrangeFrom) {
        fL = arrangeFrom.x;
        fT = arrangeFrom.y;
      }
      if (arrangeTo) {
        tL = arrangeTo.x;
        tT = arrangeTo.y;
      }
      CA   = Math.abs(tT - fT);
      CO   = Math.abs(tL - fL);
      H    = Math.sqrt(CA*CA + CO*CO);
    }

    var top  = (tT+fT)/2 - W/2 - 60;
    var left = (tL+fL)/2 - H/2;

    var arrows  = [...line.getElementsByClassName('arrow')];

    var needSwap = (fL > tL || (fL == tL && fT < tT));
    var arrowFw = needSwap && isVisible(arrows[0]) && arrows[0] || !needSwap && isVisible(arrows[1]) && arrows[1];
    var arrowBw = !needSwap && isVisible(arrows[0]) && arrows[0] || needSwap && isVisible(arrows[1]) && arrows[1];
    if (arrowFw) {
      arrowFw.classList.remove('arrow-bw');
      arrowFw.classList.add('arrow-fw');
      arrowFw.style.borderRightColor = color;
      arrowFw.style.top = W/2-6 + "px";
    }
    if (arrowBw) {
      arrowBw.classList.remove('arrow-fw');
      arrowBw.classList.add('arrow-bw');
      arrowBw.style.borderLeftColor = color;
      arrowBw.style.top = W/2-6 + "px";
    }
    line.style.display = "none";
    line.style["-webkit-transform"] = 'rotate('+ ANG +'deg)';
    line.style["-moz-transform"] = 'rotate('+ ANG +'deg)';
    line.style["-ms-transform"] = 'rotate('+ ANG +'deg)';
    line.style["-o-transform"] = 'rotate('+ ANG +'deg)';
    line.style["-transform"] = 'rotate('+ ANG +'deg)';
    line.style.top    = top+'px';
    line.style.left   = left+'px';
    line.style.width  = H + 'px';
    line.style.height = W + 'px';
    line.style.background = "linear-gradient(to right, " +
      (arrowFw ? "transparent" : color) +" 11px, " +
      color + " 11px " + (H - 11) + "px, " +
      (arrowBw ? "transparent" : color) + " " + (H - 11) + "px 100%)";
    line.style.display = "initial";
  }

  function repaintConnection(connectionElement) {
    var fromQ = connectionElement.getAttribute('from');
    var fromE = document.querySelector(fromQ);
    var toQ = connectionElement.getAttribute('to');
    var toE = document.querySelector(toQ);
    connectedObserver.observe(fromE, {attributes:true});
    connectedObserver.observe(toE, {attributes:true});

    var lineE = connectionElement.getElementsByClassName('line')[0];
    if (!lineE) {
      lineE = document.createElement('div');
      lineE.classList.add('line');
      connectionElement.appendChild(lineE);
    }
    var needTail = connectionElement.hasAttribute('tail');
    var needHead = connectionElement.hasAttribute('head');
    var arrows = lineE.getElementsByClassName('arrow');
    var tail = arrows[0];
    var head = arrows[1];
    if (!tail && (needHead || needTail)) {
      tail = document.createElement('div');
      tail.classList.add('arrow');
      lineE.appendChild(tail);
    }

    if (!head && needHead) {
      head = document.createElement('div');
      head.classList.add('arrow');
      lineE.appendChild(head);
    }

    tail && (tail.style.visibility = needTail ? 'visible' : 'hidden');
    head && (head.style.visibility = needHead ? 'visible' : 'hidden');

    var textE = lineE.getElementsByClassName('text')[0];
    var textMessage = connectionElement.getAttribute('text');

    if (!textE && textMessage) {
      textE = document.createElement('div');
      textE.classList.add('text');
      lineE.appendChild(textE);
    }
    if (textE && textE.innerText != textMessage) {
      textE.innerText = textMessage;
    }

    adjustLine(fromE, toE, lineE, {
      color: connectionElement.getAttribute('color'),
      onlyVisible: connectionElement.hasAttribute('onlyVisible'),
      fromX: parseFloat(connectionElement.getAttribute('fromX')),
      fromY: parseFloat(connectionElement.getAttribute('fromY')),
      toX: parseFloat(connectionElement.getAttribute('toX')),
      toY: parseFloat(connectionElement.getAttribute('toY')),
      width: parseFloat(connectionElement.getAttribute('width'))
    });
  }

  function repaintWithoutObserve(tag) {
    connectionObserver.observe(tag, {attributeFilter: []});
    repaintConnection(tag);
    connectionObserver.observe(tag, {attributes:true, childList:true, subtree: true});
  }

  function createOne(newElement) {
    connectionElements.push(newElement);
    repaintConnection(newElement);
    connectionObserver.observe(newElement, {attributes:true, childList:true, subtree: true});
  }

  function create() {
    bodyObserver.observe(document.body, {childList:true, subtree: true});
    [...document.body.getElementsByTagName('connection')].forEach(createOne);
  }

  function removeConnection(tag) {
    for(var i = connectionElements.length - 1; i >= 0; i--)
      if(connectionElements[i] === tag)
        connectionElements.splice(i, 1);

    connectionObserver.observe(tag, {attributeFilter: []});
  }

  function changedConnectedTag(changes) {
    changes.forEach(e => {
      var changedElem = e.target;
      var wasConnection = false;
      for (var i = 0; i < connectionElements.length; ++i) {
        var fromE = document.querySelector(connectionElements[i].getAttribute('from'));
        if (fromE === changedElem) {
          wasConnection = true;
          repaintWithoutObserve(connectionElements[i]);
          continue;
        }
        var toE = document.querySelector(connectionElements[i].getAttribute('to'));
        if (toE === changedElem) {
          wasConnection = true;
          repaintWithoutObserve(connectionElements[i]);
        }
      }
      if (!wasConnection) {
        connectionObserver.observe(changedElem, {attributeFilter: []});
      }
    });
  }

  function changedConnectionTag(changes) {
    changes.forEach(e => {
      var conn = e.target;
      if (conn.tagName.toLowerCase() !== 'connection' && e.attributeName === 'class')
        ;
      while (conn && conn.tagName.toLowerCase() !== 'connection')
        conn = conn.parentElement;
      if (!conn) return;
      repaintWithoutObserve(conn);
    });
  }

  function bodyNewElement(changes) {
    changes.forEach(e => {
      e.removedNodes.forEach(n => {
        if (n.tagName && n.tagName.toLowerCase() === 'connection')
          removeConnection(n);
      });
      e.addedNodes.forEach(n => {
        if (n.tagName && n.tagName.toLowerCase() === 'connection')
          createOne(n);
      });
    });
  }

  var connectionElements = [];
  var MutationObserver = window.MutationObserver ||
    window.WebKitMutationObserver || window.MozMutationObserver;
  var bodyObserver = new MutationObserver(bodyNewElement);
  var connectionObserver = new MutationObserver(changedConnectionTag);
  var connectedObserver = new MutationObserver(changedConnectedTag);
  document.body && create() || window.addEventListener("load", create);
})();


