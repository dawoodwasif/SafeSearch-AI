import PropTypes from 'prop-types';
import React, { useCallback, useState, useEffect } from 'react';

import TextField from '@mui/material/TextField';
import CircularProgress from '@mui/material/CircularProgress';
import Autocomplete from '@mui/material/Autocomplete';

import { RedivisButtonText, RedivisButton } from '../helpers/Buttons';

import getDataset from '../helpers/getDataset';
import listDatasets from '../helpers/listDatasets';
import listVersions from '../helpers/listVersions';
import listTables from '../helpers/listTables';
import listVariablesByTableName from '../helpers/listVariablesByTableName';
import tableHasVariable from '../helpers/tableHasVariable';

import * as styles from './styles.css';

function formatQuasiIdentifiers(variable, table, tables, variablesByTableName){
	let quasiIdentifiers = [];

	for (let datasetTable of tables){
		if (variable ? !tableHasVariable(variable, datasetTable, variablesByTableName) : table.name !== datasetTable.name) continue;

		for (const datasetTableVariable of (variablesByTableName[datasetTable.name] || [])){
			if (variable && datasetTableVariable.name.toLowerCase() === variable.name.toLowerCase()) continue;
			quasiIdentifiers.push(({ variable: datasetTableVariable, table: datasetTable }))
		}
	}

	return quasiIdentifiers;
}

export default function Settings({
	isUserAuthorized,
	onAuthorize,
	onDeauthorize,
	owner,
	setOwner,
	dataset,
	setDataset,
	datasets,
	setDatasets,
	version,
	setVersion,
	versions,
	setVersions,
	table,
	setTable,
	tables,
	setTables,
	variable,
	setVariable,
	variables,
	setVariables,
	selectedQuasiIdentifiers,
	setSelectedQuasiIdentifiers,
	variablesByTableName,
	setVariablesByTableName,
	onCalculateRisk,
}) {
	const [isDatasetsOpen, setIsDatasetsOpen] = useState(false);
	const isDatasetsLoading = isDatasetsOpen && datasets.length === 0;

	const [isVersionsOpen, setIsVersionsOpen] = useState(false);
	const isVersionsLoading = isVersionsOpen && versions.length === 0;

	const [isTablesOpen, setIsTablesOpen] = useState(false);
	const isTablesLoading = isTablesOpen && tables.length === 0;

	const [isVariablesOpen, setIsVariablesOpen] = useState(false);
	const isVariablesLoading = isVariablesOpen && variables.length === 0;

	const [filteredQuasiIdentifiers, setFilteredQuasiIdentifiers] = useState([])
	const [isQuasiIdentifiersOpen, setIsQuasiIdentifiersOpen] = useState(false);
	const isQuasiIdentifiersLoading = isQuasiIdentifiersOpen && filteredQuasiIdentifiers.length === 0;

	const [isCalculatingRisk, setIsCalculatingRisk] = useState(false);

	useEffect(() => {
		let active = true;

		if (!isDatasetsLoading) {
			return undefined;
		}

		(async () => {
			if (active && owner) {
				const nextDatasets = await listDatasets(owner);
				setDatasets(nextDatasets);
			}
		})();

		return () => {
			active = false;
		};
	}, [isDatasetsLoading]);

	useEffect(() => {
		let active = true;

		if (!isVersionsLoading) {
			return undefined;
		}

		(async () => {
			if (active && owner && dataset) {
				const nextVersions = await listVersions(dataset);
				setVersions(nextVersions);
			}
		})();

		return () => {
			active = false;
		};
	}, [isVersionsLoading]);

	const handleOwnerChange = useCallback((nextOwner) => {
		if (nextOwner !== owner){
			setOwner(nextOwner);
			setDatasets([]);
			setDataset(null);
			setTables([]);
			setTable(null);
			setVariablesByTableName({});
			setVariables([]);
			setVariable(null);
			setFilteredQuasiIdentifiers([]);
			setSelectedQuasiIdentifiers([]);
		}
	}, [owner]);

	const handleDatasetChange = useCallback((e, nextDataset) => {
		if (nextDataset?.name !== dataset?.name){
			setDataset(nextDataset);
			setTables([]);
			setTable(null);
			setVariablesByTableName({});
			setVariables([]);
			setVariable(null);
			setFilteredQuasiIdentifiers([]);
			setSelectedQuasiIdentifiers([]);
		}
	}, [dataset]);

	useEffect(() => {
		if (isUserAuthorized){
			if (owner && dataset){
				(async () => {
					const nextDataset = await getDataset(owner, dataset);
					setDataset(nextDataset);
					setVersion((nextDataset || {}).getProperty?.('currentVersion') || (nextDataset || {}).getProperty?.('nextVersion') || null);
					const nextTables = await listTables(nextDataset);
					setTables(nextTables);
				})()
			} else {
				setDataset(null);
				setVersion(null);
			}
			setVersions([]);
		}
	}, [isUserAuthorized, dataset?.name])

	const handleVersionChange = useCallback((e, nextVersion) => {
		if (nextVersion?.tag !== version?.tag){
			setVersion(nextVersion);
			setTables([]);
			setTable(null);
			setVariablesByTableName({});
			setVariables([]);
			setVariable(null);
			setFilteredQuasiIdentifiers([]);
			setSelectedQuasiIdentifiers([]);
		}
	}, [version])

	useEffect(() => {
		if (isUserAuthorized){
			if (owner && dataset && version){
				(async () => {
					let nextDataset = dataset;
					if (dataset.getProperty?.('version')?.tag !== version?.tag){
						nextDataset = await getDataset(owner, dataset, { version: version.tag })
						setDataset(nextDataset);
						const nextTables = await listTables(nextDataset);
						setTables(nextTables);
					}
				})()
			} else {
				setTables([])
			}
		}
	}, [isUserAuthorized, version])

	const handleTableChange = useCallback((e, nextTable) => {
		if (nextTable?.name !== table?.name){
			setTable(nextTable);
			setVariablesByTableName({});
			setVariables([]);
			setVariable(null);
			setFilteredQuasiIdentifiers([]);
			setSelectedQuasiIdentifiers([]);
		}
	}, [table])

	useEffect(() => {
		if (isUserAuthorized){
			if (owner && dataset && version && table && tables.length){
				(async () => {
					const nextVariablesByTableName = await listVariablesByTableName(tables);
					setVariables(nextVariablesByTableName[table.name])
					setVariablesByTableName(nextVariablesByTableName)
				})()
			} else {
				setVariables([]);
				setVariablesByTableName({});
			}
		}
	}, [isUserAuthorized, tables, table])

	useEffect(() => {
		if (table && tables.length){
			const nextFilteredQuasiIdentifiers = formatQuasiIdentifiers(variable, table, tables, variablesByTableName);
			setFilteredQuasiIdentifiers(nextFilteredQuasiIdentifiers);
		} else {
			setFilteredQuasiIdentifiers([]);
		}
	}, [variable, table, tables, variablesByTableName]);

	const handleClearAll = useCallback(() => {
		handleOwnerChange('');
	}, [handleOwnerChange])

	const handleCalculateRisk = useCallback(async () => {
		setIsCalculatingRisk(true);
		try {
			await onCalculateRisk();
		} catch (e){
			console.error(e);
		} finally {
			setIsCalculatingRisk(false);
		}
	}, [onCalculateRisk])

	return (
		<div className={styles.sideBarWrapper}>
			<div className={styles.bodyWrapper}>
				<div className={styles.sectionWrapper}>
					{isUserAuthorized ?
						(
							<div>
								<div className={styles.sectionHeader}>{'Logged in'}</div>
								<RedivisButtonText onClick={onDeauthorize}>{'Remove authentication'}</RedivisButtonText>
							</div>
						)
							:
						(
							<RedivisButton onClick={onAuthorize}>{'Authenticate with Redivis'}</RedivisButton>
						)
					}
				</div>
				{isUserAuthorized && <React.Fragment>
					<div className={styles.sectionWrapper}>
						<div className={styles.fieldWrapper}>
							<TextField
								name={'owner'}
								label={'Organization'}
								value={owner}
								onChange={(e) => handleOwnerChange(e.target.value)}
								fullWidth={true}
								variant={'outlined'}
							/>
						</div>
						<div className={styles.fieldWrapper}>
							<Autocomplete
								id={'dataset'}
								open={isDatasetsOpen}
								disabled={!owner}
								onOpen={() => {
									setIsDatasetsOpen(true);
								}}
								onClose={() => {
									setIsDatasetsOpen(false);
								}}
								value={dataset}
								onChange={handleDatasetChange}
								isOptionEqualToValue={(option, value) => option.name === value.name}
								getOptionLabel={(option) => option.name}
								options={datasets}
								loading={isDatasetsLoading}
								renderInput={(params) => (
									<TextField
										{...params}
										variant={'outlined'}
										label={'Dataset'}
										InputProps={{
											...params.InputProps,
											endAdornment: (
												<React.Fragment>
													{isDatasetsLoading ? (
														<CircularProgress color={'inherit'} size={20} />
													) : null}
													{params.InputProps.endAdornment}
												</React.Fragment>
											),
										}}
									/>
								)}
							/>
						</div>
						<div className={styles.fieldWrapper}>
							<Autocomplete
								id={'version'}
								open={isVersionsOpen}
								disabled={!owner || !dataset}
								onOpen={() => {
									setIsVersionsOpen(true);
								}}
								onClose={() => {
									setIsVersionsOpen(false);
								}}
								value={version}
								onChange={handleVersionChange}
								isOptionEqualToValue={(option, value) => option.tag === value.tag}
								getOptionLabel={(option) => option.tag}
								options={versions}
								loading={isVersionsLoading}
								renderInput={(params) => (
									<TextField
										{...params}
										variant={'outlined'}
										label={'Version'}
										InputProps={{
											...params.InputProps,
											endAdornment: (
												<React.Fragment>
													{isVersionsLoading ? (
														<CircularProgress color={'inherit'} size={20} />
													) : null}
													{params.InputProps.endAdornment}
												</React.Fragment>
											),
										}}
									/>
								)}
							/>
						</div>
						<div className={styles.fieldWrapper}>
							<Autocomplete
								id={'table'}
								open={isTablesOpen}
								disabled={!owner || !dataset || !version}
								onOpen={() => {
									setIsTablesOpen(true);
								}}
								onClose={() => {
									setIsTablesOpen(false);
								}}
								value={table}
								onChange={handleTableChange}
								isOptionEqualToValue={(option, value) => option.name === value.name}
								getOptionLabel={(option) => option.name}
								options={tables}
								loading={isTablesLoading}
								renderInput={(params) => (
									<TextField
										{...params}
										variant={'outlined'}
										label={'Primary input table'}
										InputProps={{
											...params.InputProps,
											endAdornment: (
												<React.Fragment>
													{isTablesLoading ? (
														<CircularProgress color={'inherit'} size={20} />
													) : null}
													{params.InputProps.endAdornment}
												</React.Fragment>
											),
										}}
									/>
								)}
							/>
						</div>
					</div>
					<div className={styles.sectionWrapper}>
						<div className={styles.fieldWrapper}>
							<Autocomplete
								id={'variable'}
								open={isVariablesOpen}
								disabled={!owner || !dataset || !version || !table}
								onOpen={() => {
									setIsVariablesOpen(true);
								}}
								onClose={() => {
									setIsVariablesOpen(false);
								}}
								value={variable}
								onChange={(e, newValue) => setVariable(newValue)}
								isOptionEqualToValue={(option, value) => option.name === value.name}
								getOptionLabel={(option) => option.name}
								options={variables}
								loading={isVariablesLoading}
								renderInput={(params) => (
									<TextField
										{...params}
										variant={'outlined'}
										label={'Entity ID'}
										InputProps={{
											...params.InputProps,
											endAdornment: (
												<React.Fragment>
													{isVariablesLoading ? (
														<CircularProgress color={'inherit'} size={20} />
													) : null}
													{params.InputProps.endAdornment}
												</React.Fragment>
											),
										}}
									/>
								)}
							/>
						</div>
						<div className={styles.fieldWrapper}>
							<Autocomplete
								multiple
								id={'quasiIdentifier'}
								open={isQuasiIdentifiersOpen}
								disabled={!owner || !dataset || !version || !table}
								onOpen={() => {
									setIsQuasiIdentifiersOpen(true);
								}}
								onClose={() => {
									setIsQuasiIdentifiersOpen(false);
								}}
								value={selectedQuasiIdentifiers}
								onChange={(e, newValue) => setSelectedQuasiIdentifiers(newValue)}
								isOptionEqualToValue={(option, value) => `${option.table.name}_${option.variable.name}` === `${value.table.name}_${value.variable.name}`}
								getOptionLabel={(option) => `${option.variable.name} in ${option.table.name}`}
								options={filteredQuasiIdentifiers}
								loading={isQuasiIdentifiersLoading}
								renderInput={(params) => (
									<TextField
										{...params}
										variant={'outlined'}
										label={'Quasi-identifiers'}
										InputProps={{
											...params.InputProps,
											endAdornment: (
												<React.Fragment>
													{isQuasiIdentifiersLoading ? (
														<CircularProgress color={'inherit'} size={20} />
													) : null}
													{params.InputProps.endAdornment}
												</React.Fragment>
											),
										}}
									/>
								)}
							/>
						</div>
						<div className={styles.fieldWrapper}>
							<RedivisButton type={'secondary'} onClick={handleClearAll} className={styles.clearAll}>{'Clear all'}</RedivisButton>
						</div>
					</div>
					<div className={styles.sectionWrapper}>
						<RedivisButton
							hasLoading={true}
							disabled={!owner || !dataset || !version || !table || !selectedQuasiIdentifiers?.length}
							loading={isCalculatingRisk}
							onClick={handleCalculateRisk}
						>
							{'Calculate re-identification risk'}
						</RedivisButton>
					</div>
				</React.Fragment>}
			</div>
		</div>
	);
}

Settings.propTypes = {
	isUserAuthorized: PropTypes.bool,
	onAuthorize: PropTypes.func,
	onDeauthorize: PropTypes.func,
	owner: PropTypes.string,
	setOwner: PropTypes.func,
	dataset: PropTypes.object,
	setDataset: PropTypes.func,
	datasets: PropTypes.arrayOf(PropTypes.object),
	setDatasets: PropTypes.func,
	version: PropTypes.object,
	setVersion: PropTypes.func,
	versions: PropTypes.arrayOf(PropTypes.object),
	setVersions: PropTypes.func,
	table: PropTypes.object,
	setTable: PropTypes.func,
	tables: PropTypes.arrayOf(PropTypes.object),
	setTables: PropTypes.func,
	variable: PropTypes.object,
	setVariable: PropTypes.func,
	variables: PropTypes.arrayOf(PropTypes.object),
	setVariables: PropTypes.func,
	selectedQuasiIdentifiers: PropTypes.arrayOf(PropTypes.object),
	setSelectedQuasiIdentifiers: PropTypes.func,
	variablesByTableName: PropTypes.object,
	setVariablesByTableName: PropTypes.func,
	onCalculateRisk: PropTypes.func,
};
