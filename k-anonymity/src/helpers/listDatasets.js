import { organization } from 'redivis';

const organizationsCache = {};

export default async function listDatasets(shortName) {
	const preloadedDatasets = organizationsCache[shortName.toLowerCase()]
	if (preloadedDatasets){
		return preloadedDatasets;
	} else {
		let loadedDatasets = [];
		try {
			const response = await organization(shortName).listDatasets();
			loadedDatasets = response;
			organizationsCache[shortName.toLowerCase()] = loadedDatasets;
		} catch (e){
			console.error(e)
		}
		return loadedDatasets;
	}
}
