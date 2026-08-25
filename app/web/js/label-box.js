/*
 * Drawing boxes on an image, which is the thing a user could not do in this product.
 *
 * ai.custom_training has had the trainers, the split logic and the refusals for a while.
 * What it did not have was any way to produce a label without already owning YOLO text
 * files -- so "train your own model" meant "bring your own corpus", and the registry row
 * said so rather than pretending otherwise.
 *
 * Two decisions worth stating, because both are places this could quietly go wrong.
 *
 * Coordinates are normalised to 0..1 against the IMAGE, not the canvas. A box stored in
 * screen pixels is wrong the moment the window is resized or the same corpus is opened
 * on another monitor, and every trainer here resizes its input anyway. The conversion
 * happens once, on commit, against the natural dimensions.
 *
 * A drag that ends where it started is discarded rather than stored as a zero-area box.
 * That is almost always a click -- selecting the image, dismissing something -- and a
 * degenerate box written to a label file becomes a target no model can ever match.
 */

(function (global) {
  'use strict';

  const MIN_DRAG_PX = 4;

  function clamp01(value) {
    if (value < 0) return 0;
    if (value > 1) return 1;
    return value;
  }

  class BoxLabeller {
    /**
     * @param {HTMLCanvasElement} canvas  where the image and boxes are drawn
     * @param {Object} options
     *   classes: string[]   the label palette; the first is selected initially
     *   onChange: function  called after every commit or removal
     */
    constructor(canvas, options) {
      const settings = options || {};
      this.canvas = canvas;
      this.context = canvas.getContext('2d');
      this.classes = (settings.classes || []).slice();
      if (!this.classes.length) {
        throw new Error('A labeller needs at least one class; boxes cannot be anonymous.');
      }
      this.activeClass = this.classes[0];
      this.onChange = settings.onChange || function () {};
      this.boxes = [];
      this.image = null;
      this.drag = null;
      this._bind();
    }

    _bind() {
      const canvas = this.canvas;
      canvas.addEventListener('pointerdown', (event) => this._start(event));
      canvas.addEventListener('pointermove', (event) => this._move(event));
      canvas.addEventListener('pointerup', (event) => this._finish(event));
      // A pointer that leaves the canvas mid-drag would otherwise leave the labeller
      // convinced a drag is still in progress, and the next click would commit a box
      // spanning wherever the pointer went.
      canvas.addEventListener('pointercancel', () => { this.drag = null; this.draw(); });
    }

    setImage(image) {
      this.image = image;
      this.canvas.width = image.naturalWidth || image.width;
      this.canvas.height = image.naturalHeight || image.height;
      this.draw();
    }

    setActiveClass(name) {
      if (this.classes.indexOf(name) === -1) {
        throw new Error('Unknown class: ' + name);
      }
      this.activeClass = name;
    }

    _point(event) {
      const rect = this.canvas.getBoundingClientRect();
      // The canvas is displayed at whatever size CSS gives it, which is rarely its
      // pixel size. Scaling by that ratio is what keeps a box under the cursor.
      const scaleX = this.canvas.width / (rect.width || 1);
      const scaleY = this.canvas.height / (rect.height || 1);
      return {
        x: (event.clientX - rect.left) * scaleX,
        y: (event.clientY - rect.top) * scaleY,
      };
    }

    _start(event) {
      const point = this._point(event);
      this.drag = { x0: point.x, y0: point.y, x1: point.x, y1: point.y };
    }

    _move(event) {
      if (!this.drag) return;
      const point = this._point(event);
      this.drag.x1 = point.x;
      this.drag.y1 = point.y;
      this.draw();
    }

    _finish(event) {
      if (!this.drag) return;
      const point = this._point(event);
      this.drag.x1 = point.x;
      this.drag.y1 = point.y;
      const drag = this.drag;
      this.drag = null;

      const width = Math.abs(drag.x1 - drag.x0);
      const height = Math.abs(drag.y1 - drag.y0);
      if (width < MIN_DRAG_PX || height < MIN_DRAG_PX) {
        // A click, not a box. Discarded rather than stored with zero area.
        this.draw();
        return;
      }
      const left = Math.min(drag.x0, drag.x1);
      const top = Math.min(drag.y0, drag.y1);
      this.boxes.push({
        label: this.activeClass,
        x: clamp01(left / this.canvas.width),
        y: clamp01(top / this.canvas.height),
        width: clamp01(width / this.canvas.width),
        height: clamp01(height / this.canvas.height),
      });
      this.draw();
      this.onChange(this.boxes.slice());
    }

    removeBox(index) {
      if (index < 0 || index >= this.boxes.length) return false;
      this.boxes.splice(index, 1);
      this.draw();
      this.onChange(this.boxes.slice());
      return true;
    }

    clear() {
      this.boxes = [];
      this.draw();
      this.onChange([]);
    }

    colourFor(label) {
      // Stable per label rather than per index, so a class keeps its colour when
      // another class is added.
      let hash = 0;
      for (let i = 0; i < label.length; i += 1) {
        hash = (hash * 31 + label.charCodeAt(i)) % 360;
      }
      return 'hsl(' + hash + ', 80%, 55%)';
    }

    draw() {
      const context = this.context;
      if (!context) return;
      context.clearRect(0, 0, this.canvas.width, this.canvas.height);
      if (this.image) {
        context.drawImage(this.image, 0, 0, this.canvas.width, this.canvas.height);
      }
      context.lineWidth = Math.max(2, this.canvas.width / 300);
      this.boxes.forEach((box) => {
        context.strokeStyle = this.colourFor(box.label);
        context.strokeRect(
          box.x * this.canvas.width,
          box.y * this.canvas.height,
          box.width * this.canvas.width,
          box.height * this.canvas.height
        );
      });
      if (this.drag) {
        context.strokeStyle = this.colourFor(this.activeClass);
        context.setLineDash([6, 4]);
        context.strokeRect(
          Math.min(this.drag.x0, this.drag.x1),
          Math.min(this.drag.y0, this.drag.y1),
          Math.abs(this.drag.x1 - this.drag.x0),
          Math.abs(this.drag.y1 - this.drag.y0)
        );
        context.setLineDash([]);
      }
    }

    /**
     * What gets posted to the API. `confirmed_empty` is carried explicitly because the
     * builder refuses an unlabelled image: an image with no boxes is either a deliberate
     * negative or one someone forgot, and only the user knows which.
     */
    toPayload(imagePath, confirmedEmpty) {
      return {
        path: imagePath,
        boxes: this.boxes.slice(),
        confirmed_empty: Boolean(confirmedEmpty) && this.boxes.length === 0,
      };
    }
  }

  const api = { BoxLabeller: BoxLabeller, MIN_DRAG_PX: MIN_DRAG_PX };
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
  global.ODKLabelBox = api;
})(typeof window !== 'undefined' ? window : globalThis);
