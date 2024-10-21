import { query } from 'redivis';

import tableHasVariable from '../helpers/tableHasVariable';

const SAMPLE_PERCENT = 1;

function escape(name){
	return name.replace(/\W+/g, '_');
}

function getRiskQuery(variable, quasiIdentifiers, variablesByTableName, table, tables, dataset, organizationIdentifier){
	if (quasiIdentifiers.length === 0) return null;
	const persistentVariableName = variable.name;

	const matchedTables = tables.filter((matchedTable) => {
		if (table.name === matchedTable.name) return false;
		if (tableHasVariable(variable, matchedTable, variablesByTableName)) return true;
		return false;
	})

	const filteredMatchedTables = matchedTables.filter((matchedTable) => {
		return quasiIdentifiers.some((quasiIdentifier) =>
			quasiIdentifier.table.name === matchedTable.name
		);
	})

	const INT64_MIN = (2 ** 64 / 2) * -1;

	const qualifiedDataset = `${escape(organizationIdentifier)}.${escape(dataset.name)}`
	const escapedTableIdentifier = escape(table.name);

	const identifiersForTable = quasiIdentifiers.filter((quasiIdentifier) => quasiIdentifier.table.name === table.name);

	const sampleClause = `
		WHERE FARM_FINGERPRINT(CAST(${persistentVariableName} AS STRING)) < ${INT64_MIN + 2 ** 64 / 100 * SAMPLE_PERCENT}
	`;

	return `
		SELECT
			SUM(group_size) record_count,
			CASE 
				WHEN group_size <= 10 THEN CAST(group_size AS STRING)
				WHEN group_size >10 AND group_size <= 15 THEN '(10,15]'
				WHEN group_size >15 AND group_size <= 20 THEN '(15,20]'
				ELSE '>20' 
			END as bucket
		FROM (
		SELECT
			COUNT(DISTINCT ${persistentVariableName}) AS group_size ${identifiersForTable.length + quasiIdentifiers.length ? ',' : ''}
			${
				quasiIdentifiers
					.map(({ table, variable }) => `${escape(table.name)}.${variable.name}`)
					.join(', ')
			} 
		  
			FROM (
				SELECT 
					${persistentVariableName} ${identifiersForTable.length ? ', ' + identifiersForTable.map(({ variable }) => variable.name).join(', ') : ''}
				FROM ${qualifiedDataset}.${escapedTableIdentifier}
				${sampleClause}
				GROUP BY 
					${persistentVariableName} ${identifiersForTable.length ? ', ' +  identifiersForTable.map(({ variable }) => variable.name).join(', ') : ''}
			) AS ${escapedTableIdentifier}
		  
			${filteredMatchedTables
				.map(({ name }) => {
					const identifiersForTable = quasiIdentifiers.filter(({ table }) => escape(name) === escape(table.name));
					return `
						LEFT JOIN (
							SELECT 
								${persistentVariableName},  ${identifiersForTable.map(({ variable }) => variable.name).join(', ')}
							FROM ${qualifiedDataset}.${escape(name)}
							${sampleClause}
							GROUP BY 
								${persistentVariableName}, ${identifiersForTable.map(({ variable }) => variable.name).join(', ')}
						) AS ${escape(name)} 
						USING(${persistentVariableName})
					`;
				})
				.join('\n')
			}
		  
			GROUP BY ${
				quasiIdentifiers
					.map(({ table, variable }) => `${escape(table.name)}.${variable.name}`)
				.join(',')
			}
	)t
	  
	  GROUP BY bucket
	`;
}

const sampleData = [
	{ bucket: 5, record_count: 1000 },
	{ bucket: 6, record_count: 1000 },
	{ bucket: 7, record_count: 1000 },
	{ bucket: 8, record_count: 1000 },
	{ bucket: 9, record_count: 1000 },
]

export default async function computeRisk(variable, quasiIdentifiers, variablesByTableName, table, tables, dataset, organizationIdentifier) {
	// return sampleData;
	const riskQuery = getRiskQuery(variable, quasiIdentifiers, variablesByTableName, table, tables, dataset, organizationIdentifier);
	console.log('--riskQuery--\n', riskQuery);
	const queryResponse = await query(riskQuery).listRows();
	return queryResponse;
}
