import PropTypes from 'prop-types';
import React, { useRef, useEffect, useCallback, useState } from 'react';

import GraphRenderer from './GraphRenderer';

import * as styles from './styles.css';

const DEFAULT_BUCKET = 10;

export default function Graph({ queryResponse, dataset }) {
	const graphElement = useRef(null);
	const graphRenderer = useRef(null);

	const [highlightedBucket, setHighlightedBucket] = useState(DEFAULT_BUCKET);

	useEffect(() => {
		if (graphElement.current) {
			if (!graphRenderer.current) {
				graphRenderer.current = new GraphRenderer(graphElement.current, {
					data: queryResponse,
					highlightedBucket,
					setHighlightedBucket,
				});
			} else {
				graphRenderer.current.update({
					data: queryResponse,
					highlightedBucket,
					setHighlightedBucket,
				});
			}
		}
		return () => {
			if (graphRenderer.current) {
				graphRenderer.current.unload();
			}
		};
	}, [
		queryResponse,
		highlightedBucket,
	]);

	const renderGraphText = useCallback(() => {
		const droppedRecordCount = queryResponse.reduce((totalRows, { bucket, record_count }) => totalRows += isNaN(bucket) || bucket > highlightedBucket ? 0 : record_count, 0);
		const totalRecordCount = queryResponse.reduce((totalRows, { record_count }) => totalRows += record_count, 0);
		const percent = Math.round(droppedRecordCount / totalRecordCount * 1e7) / 1e5;

		return (
			<>
				<p>{`${percent}% of this table's total records (${totalRecordCount}) have a k-anonymity value less than or equal to ${highlightedBucket}`}</p>
				<br/>
				<p>{`A k-anonymity value of ${highlightedBucket} means that a given entity is indistinguishable from at least ${highlightedBucket - 1} other values, given the quasi-identifiers provided.`}</p>
				<br/>
				<p>{`This table's lowest k-anonymity value is ${Math.min(...queryResponse.map(({bucket}) => isNaN(bucket) ? Infinity : bucket))}`}</p>
			</>
		);

	}, [queryResponse, dataset, highlightedBucket])

	return (
		<div className={styles.graphWrapper}>
			{!queryResponse &&
				<div className={styles.descriptionWrapper}>
					<p>{`This tool can help measure the re-identification risk of the primary entity in this dataset. It will determine to what degree individuals are anonymized based on their uniqueness within the dataset.`}</p>
				</div>
			}
			{queryResponse &&
				<>
					<div className={styles.header}>{`K-anonymity in ${dataset?.name}`}</div>
					<div ref={graphElement} className={styles.wrapper} />
					<div className={styles.footer}>{renderGraphText()}</div>
				</>
			}
		</div>
	);
}

Graph.propTypes = {
	queryResponse: PropTypes.arrayOf(PropTypes.object),
	dataset: PropTypes.object,
};