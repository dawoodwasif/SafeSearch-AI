import React, { useEffect, useState, useCallback } from 'react';
import { withRouter } from 'react-router-dom';

import Graph from '../Graph';
import Settings from '../Settings';
import Header from '../Header';
import { createTheme, ThemeProvider } from '@mui/material/styles';

import computeRisk from '../helpers/computeRisk';

import * as styles from './styles.css';
import {
	authorize,
	deauthorize,
	isAuthorized,
} from 'redivis';

const DEFAULT_OWNER = 'Demo';
const DEFAULT_DATASET = { name: 'CMS 2014 Medicare Data' };
const DEFAULT_VERSION = null;
const DEFAULT_TABLE = { name: 'Inpatient charges' };
const DEFAULT_VARIABLE = { name: 'provider_id' };
const DEFAULT_VARIABLES_BY_TABLE_NAME = {
	'Inpatient charges': [DEFAULT_VARIABLE]
};
const DEFAULT_SELECTED_QUASI_IDENTIFIERS = [{ variable: { name: 'provider_state' }, table: DEFAULT_TABLE }];

const theme = createTheme({
	palette: {
		primary: {
			light: '#a88bc9',
			main: '#7f3b97',
			dark: '#5e1e75',
			contrastText: '#ffffff',
		},
		secondary: {
			light: '#d4d3db',
			main: '#4d4c54',
			dark: '#353538',
			contrastText: '#ffffff',
		},
	},
	shape: {
		borderRadius: 2,
	},
	typography: {
		fontFamily: 'Nunito Sans, sans-serif',
		fontWeightLight: 600,
		fontWeightRegular: 600,
		fontWeightMedium: 600,
		fontWeigthBold: 700,
	}
});

function App({ history }) {
	useEffect(() => {
		const path = localStorage.getItem('path');
		if (path) {
			localStorage.removeItem('path');
			history.replace(`/${path}`);
		}
	}, []);

	const [queryResponse, setQueryResponse] = useState(null);

	const [isUserAuthorized, setIsUserAuthorized] = useState(false);

	const handleSetIsUserAuthorized = useCallback(
		async () => {
			const nextIsUserAuthorized = await isAuthorized();
			setIsUserAuthorized(nextIsUserAuthorized);
		},
		[],
	);

	useEffect(() => {
		handleSetIsUserAuthorized();
	}, []);

	const handleAuthorize = useCallback(async () => {
		await authorize();
		handleSetIsUserAuthorized();
	}, [authorize]);

	const handleUnauthorize = useCallback(async () => {
		await deauthorize()
		handleSetIsUserAuthorized();
		setQueryResponse(null);
	}, []);

	const [owner, setOwner] = useState(DEFAULT_OWNER);
	const [dataset, setDataset] = useState(DEFAULT_DATASET);
	const [version, setVersion] = useState(DEFAULT_VERSION);
	const [table, setTable] = useState(DEFAULT_TABLE);
	const [variable, setVariable] = useState(DEFAULT_VARIABLE);

	const [datasets, setDatasets] = useState([]);
	const [versions, setVersions] = useState([]);
	const [tables, setTables] = useState([]);
	const [variables, setVariables] = useState([]);

	const [selectedQuasiIdentifiers, setSelectedQuasiIdentifiers] = useState(DEFAULT_SELECTED_QUASI_IDENTIFIERS);
	const [variablesByTableName, setVariablesByTableName] = useState(DEFAULT_VARIABLES_BY_TABLE_NAME);

	const handleCalculateRisk = useCallback(async () => {
		const nextQueryResponse = await computeRisk(variable, selectedQuasiIdentifiers, variablesByTableName, table, tables, dataset, owner);
		setQueryResponse(nextQueryResponse);
	}, [variable, selectedQuasiIdentifiers, variablesByTableName, table, tables, dataset, owner])

	const renderBody = () => {
		return (
				<div className={styles.bodyWrapper}>
					<div className={styles.graphWrapper}>
						<div className={styles.settings}>
							<Settings
								isUserAuthorized={isUserAuthorized}
								onAuthorize={handleAuthorize}
								onDeauthorize={handleUnauthorize}
								owner={owner}
								setOwner={setOwner}
								dataset={dataset}
								setDataset={setDataset}
								datasets={datasets}
								setDatasets={setDatasets}
								version={version}
								setVersion={setVersion}
								versions={versions}
								setVersions={setVersions}
								table={table}
								setTable={setTable}
								tables={tables}
								setTables={setTables}
								variable={variable}
								setVariable={setVariable}
								variables={variables}
								setVariables={setVariables}
								selectedQuasiIdentifiers={selectedQuasiIdentifiers}
								setSelectedQuasiIdentifiers={setSelectedQuasiIdentifiers}
								variablesByTableName={variablesByTableName}
								setVariablesByTableName={setVariablesByTableName}
								onCalculateRisk={handleCalculateRisk}
							/>
						</div>
						<div className={styles.graph}>
							<Graph
								queryResponse={queryResponse}
								dataset={dataset}
							/>
						</div>
					</div>
				</div>
		);
	};

	return (
		<ThemeProvider theme={theme}>
			<div className={styles.appWrapper}>
				<Header title={'Re-identification risk assessment'} />
				{renderBody()}
			</div>
		</ThemeProvider>
	);
}
export default withRouter(App);
