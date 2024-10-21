export default async function listTables(dataset) {
	let loadedTables = [];
	try {
		const response = await dataset.listTables();
		loadedTables = response;
	} catch (e){
		console.error(e)
	}
	return loadedTables;
}