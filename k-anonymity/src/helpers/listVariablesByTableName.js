export default async function listVariablesByTableName(tables) {
	let loadedVariablesByTableName = {};
	try {
		const variableArrays = await Promise.all(tables.map((datasetTable) => datasetTable.listVariables()));
		for (let i = 0; i < variableArrays.length; i++){
			loadedVariablesByTableName[tables[i].name] = variableArrays[i];
		}
	} catch (e){
		console.error(e)
	}
	return loadedVariablesByTableName;
}
