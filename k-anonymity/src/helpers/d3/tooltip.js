import { select as d3Select, selection as d3Selection } from 'd3-selection';

import styles from './d3.css';

export default function tooltip() {
	let node = initNode();
	const nodel = d3Select(node);
	let direction;
	let offset = [5, 0];
	let html = '';
	let container = document.body;
	let delay = 0;
	let timeout;
	let drawn = false;
	let defaultWrapper;

	document.body.appendChild(node);

	function tip(wrapper) {
		defaultWrapper = wrapper;
	}

	tip.show = function (d, wrapper = defaultWrapper) {
		clearTimeout(timeout);
		timeout = setTimeout(
			() => {
				drawn = true;

				let i = directions.length;

				nodel.html(html ? html.call(this, d) : html).style('display', 'block');

				while (i--) {
					nodel.classed(directions[i], false);
				}

				const dir = direction || getDirection(this, wrapper);
				const coords = directionCallbacks[dir].apply(this);

				nodel.classed(dir, true).style('top', `${coords.top}px`).style('left', `${coords.left}px`);

				nodel.attr('data-arrow-offset', coords.arrowOffset);
			},
			drawn ? 0 : delay,
		);

		return tip;
	};

	tip.hide = function () {
		clearTimeout(timeout);
		timeout = setTimeout(() => {
			drawn = false;
			nodel.style('display', null);
		}, delay);

		return tip;
	};

	tip.container = function (elem) {
		if (elem) {
			container = elem;
		} else {
			return container;
		}
	};

	tip.delay = function (value) {
		if (value) {
			delay = value;
		} else {
			return delay;
		}
		return tip;
	};

	tip.attr = function (name, value) {
		if (name === 'class') {
			if (value) {
				nodel.classed(value, true);
			}
		} else {
			d3Selection.prototype.attr.call(nodel, name, value);
		}

		return tip;
	};

	tip.style = function (...args) {
		if (args.length < 2 && typeof args[0] === 'string') {
			return nodel.style(args[0]);
		} else {
			d3Selection.prototype.style.apply(nodel, args);
		}

		return tip;
	};

	tip.direction = function (v) {
		if (v === undefined) return direction;
		direction = v;

		return tip;
	};

	tip.offset = function (v) {
		if (v === undefined) return offset;
		offset = v;

		return tip;
	};

	tip.html = function (v) {
		if (!arguments.length) return html;
		html = v;

		return tip;
	};

	tip.destroy = function () {
		if (node) {
			nodel.remove();
			node = null;
		}
		return tip;
	};

	const arrowHeight = 11;

	const directionCallbacks = {
		n: function () {
			// TODO: not implemented
			const bbox = this.getBoundingClientRect();
			return {
				top: bbox.top - node.offsetHeight - arrowHeight,
				left: bbox.left - node.offsetWidth / 2 + bbox.width / 2 + 17,
			};
		},
		s: function () {
			// TODO: not implemented
			const bbox = this.getBoundingClientRect();
			return {
				top: bbox.top + bbox.height + arrowHeight,
				left: bbox.left - node.offsetWidth + bbox.width / 2 + 17,
			};
		},
		e: function () {
			const bbox = this.getBoundingClientRect();
			return {
				top: bbox.top + bbox.height / 2 - node.offsetHeight / 2,
				left: bbox.left + bbox.width / 2,
			};
		},
		w: function () {
			const bbox = this.getBoundingClientRect();

			return {
				top: bbox.top + bbox.height / 2 - node.offsetHeight / 2,
				left: bbox.left - node.offsetWidth + bbox.width / 2,
			};
		},
		nw: function () {
			const bbox = this.getBoundingClientRect();
			return {
				top: bbox.top - node.offsetHeight,
				left: bbox.left - node.offsetWidth + bbox.width / 2,
			};
		},
		ne: function () {
			const bbox = this.getBoundingClientRect();
			return {
				top: bbox.top - node.offsetHeight,
				left: bbox.left + bbox.width / 2,
			};
		},
		sw: function () {
			const bbox = this.getBoundingClientRect();
			return {
				top: bbox.top + bbox.height + arrowHeight,
				left: bbox.left - node.offsetWidth + bbox.width / 2,
			};
		},
		se: function () {
			const bbox = this.getBoundingClientRect();
			return {
				top: bbox.top + bbox.height + arrowHeight,
				left: bbox.left + bbox.width / 2,
			};
		},
	};

	const directions = Object.keys(directionCallbacks);

	function initNode() {
		const node = d3Select(document.createElement('div'));

		node.classed(styles.tooltip, true);

		return node.node();
	}

	function getDirection(elem, wrapper) {
		const elemBox = elem.getBoundingClientRect();
		const containerBox = container.getBoundingClientRect();

		if (elemBox.left + elemBox.width / 2 > containerBox.width / 2) {
			if (elemBox.top + elemBox.height / 2 > containerBox.height / 2) {
				return 'nw';
			} else {
				return 'w';
			}
		} else {
			if (elemBox.top + elemBox.height / 2 > containerBox.height / 2) {
				return 'ne';
			} else {
				return 'e';
			}
		}
	}

	return tip;
}
