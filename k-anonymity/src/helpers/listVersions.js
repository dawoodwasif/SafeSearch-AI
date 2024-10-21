export default async function listVersions(dataset) {
	let loadedVersions = [];
	try {
		const response = await dataset.listVersions();
		loadedVersions = response;
	} catch (e){
		console.error(e)
	}
	return loadedVersions;
}
