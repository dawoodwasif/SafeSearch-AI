import * as styles from './styles.css';

import tooltip from '../helpers/d3/tooltip';

import { select as d3Select } from 'd3-selection';
import { scaleLinear as d3ScaleLinear, scaleBand as d3ScaleBand } from 'd3-scale';
import { axisBottom as d3AxisBottom, axisRight as d3AxisRight } from 'd3-axis';

function percentage(d){
	return Math.round(d.cumulativeRecords / d.totalRecords * 100);
}

function formatData(data){
	const totalRecords = data.reduce((totalRows, { bucket, record_count }) => totalRows += record_count, 0);

	const formattedBuckets = [];
	let cumulativeRecords = 0;
	for (let i = 1; i <= 20; i++) {
		let relevantBucket = data.find(({ bucket }) => parseInt(bucket) === i);
		if (relevantBucket) cumulativeRecords += relevantBucket.record_count;
		formattedBuckets.push({ bucket: i, cumulativeRecords, totalRecords });
		if (cumulativeRecords >= totalRecords && i >= 10) break;
	}

	return formattedBuckets;
}

const barTooltip = tooltip()
	.attr('class', styles.tooltip)
	.html((d) => {
		return `
			<div class="${styles.row} ${styles.top}">
				<div class="${styles.label}">k-anonymity</div>
				<div class="${styles.value}">${d.bucket}</div>
			</div>
			<div class="${styles.divider}"></div>
			<div class="${styles.row}">
				<div class="${styles.label}">Records (cumulative)</div>
				<div class="${styles.value}">${percentage(d)}%</div>
			</div>
		`;
	});

function getScales(elem, margin, domain) {
	if (!domain) {
		return null;
	}

	const width = elem.offsetWidth;
	const height = elem.offsetHeight;

	const { left, right, top, bottom } = margin;

	const xRange = width - left - right;
	const yRange = height - top - bottom;

	const x = d3ScaleBand().rangeRound([0, xRange]).domain(domain.x).padding(0.2);
	const y = d3ScaleLinear().rangeRound([yRange, 0]).domain(domain.y);

	return { x, y };
}

function drawBars(elem, margin, scales, data, highlightedBucket, setHighlightedBucket) {
	const height = elem.offsetHeight;

	const { top, bottom } = margin;

	const yRange = height - top - bottom;

	const g = d3Select(elem).select(`.${styles.bars}`);

	const bar = g.selectAll(`.${styles.bar}`).data(data);

	const mergedBars = bar
		.enter()
		.append('rect')
		.merge(bar)
		.attr('class', (d) => `${styles.bar} ${d.bucket === highlightedBucket ? styles.highlighted : ''}`)
		.attr('x', (d) => scales.x(d.bucket))
		.attr('y', (d) => {
			return scales.y(percentage(d)) < yRange ? scales.y(percentage(d)) : yRange - 2;
		})
		.attr('width', scales.x.bandwidth())
		.attr('height', (d) => {
			return Math.max(yRange - scales.y(percentage(d)), 2);
		});

	const barOverlayEntry = g.selectAll(`.${styles.barOverlay}`).data(data).enter().append('rect');

	barOverlayEntry
		.attr('class', styles.barOverlay)
		.on('mouseenter', function (e, d) {
			const nodes = barOverlayEntry.nodes();
			const i = nodes.indexOf(this);
			const siblingBar = mergedBars.filter((d, index) => i === index);
			siblingBar.node().classList.add(styles.hover);
			barTooltip.show.call(siblingBar.node(), d, elem);
		})
		.on('mouseleave', function () {
			const nodes = barOverlayEntry.nodes();
			const i = nodes.indexOf(this);
			const siblingBar = mergedBars.filter((d, index) => i === index);
			siblingBar.node().classList.remove(styles.hover);
			barTooltip.hide.call(siblingBar.node());
		})
		.on('click', (e, d) => setHighlightedBucket(d.bucket));

	g.selectAll(`.${styles.barOverlay}`)
		.attr('x', (d) => scales.x(d.bucket))
		.attr('y', 0)
		.attr('width', scales.x.bandwidth())
		.attr('height', yRange);

	bar.exit().remove();
}

export default class GraphRenderer {
	constructor(elem, props) {
		this.elem = elem;
		this.margin = {
			top: 10,
			bottom: 60,
			left: 0,
			right: 70,
		};
		window.addEventListener('resize', () => {
			this.onResize();
		});

		const xOffset = this.margin.left;
		const yOffset = this.margin.top;

		this.svg = d3Select(elem).append('svg').attr('width', `100%`).attr('height', `100%`);

		this.svg.append('rect').attr('class', styles.wrapperRect);

		this.svg.append('g').attr('class', styles.bars).attr('transform', `translate(${xOffset}, ${yOffset})`);

		this.svg.append('g').attr('class', `${styles.axis} ${styles.xAxis}`);
		this.svg.append('g').attr('class', `${styles.axis} ${styles.yAxis}`);

		this.svg
			.append('text')
			.attr('class', `${styles.axisLabel} ${styles.xAxis}`)
			.attr('x', this.margin.left + (this.elem.offsetWidth - this.margin.left - this.margin.right)/2)
			.attr('y', this.elem.offsetHeight - this.margin.bottom + 45)
			.text('k-anonymity values');

		this.svg
			.append('text')
			.attr('class', `${styles.axisLabel} ${styles.yAxis}`)
			.attr('x', this.elem.offsetWidth - this.margin.right + 50)
			.attr('y', this.margin.top + (this.elem.offsetHeight - this.margin.bottom - this.margin.top) / 2)
			.attr('transform', `
				rotate(-90, ${this.elem.offsetWidth - this.margin.right + 50},${this.margin.top + (this.elem.offsetHeight - this.margin.bottom - this.margin.top) / 2})
			`)
			.text('cumulative % of records');

		this.svg.call(barTooltip);

		this.update(props);
	}

	onResize() {
		if (this.chart) {
			const defaultWidth = 700;
			const defaultHeight = 467;
			const height = this.elem.offsetHeight;
			const width = this.elem.offsetWidth;
			const minRatio = Math.min(height / defaultHeight, width / defaultWidth);
			this.chart.resize({ width, height, minRatio });
		}
	}

	update({ data, highlightedBucket, setHighlightedBucket }) {
		if (data) {
			this.elem.classList.add(styles.loading);

			const formattedData = formatData(data);
			console.log('formatted (CDF)', formattedData)
			if (!formattedData) return;

			const xDomain = formattedData.map((d) => d.bucket);
			const yDomain = [0, 100];
			const scales = getScales(this.elem, this.margin, { x: xDomain, y: yDomain });

			d3Select(this.elem)
				.select(`.${styles.axis}.${styles.xAxis}`)
				.call(d3AxisBottom(scales.x).tickFormat((bucket) => `${bucket}${bucket === xDomain[xDomain.length - 1] ? '+' : ''}`))
				.attr('transform', `translate(${this.margin.left}, ${this.elem.offsetHeight - this.margin.bottom})`);

			d3Select(this.elem)
				.select(`.${styles.axis}.${styles.yAxis}`)
				.call(d3AxisRight(scales.y).ticks(3))
				.attr('transform', `translate(${this.elem.offsetWidth - this.margin.right}, ${this.margin.top})`);

			d3Select(this.elem)
				.select(`.${styles.wrapperRect}`)
				.attr('transform', `translate(${this.margin.left}, ${this.margin.top})`)
				.attr('width', this.elem.offsetWidth - this.margin.left - this.margin.right)
				.attr('height', this.elem.offsetHeight - this.margin.top - this.margin.bottom);

			drawBars(this.elem, this.margin, scales, formattedData, highlightedBucket, setHighlightedBucket);


			this.onResize();
			this.elem.classList.remove(styles.loading);
		}
	}

	unload() {}
}
