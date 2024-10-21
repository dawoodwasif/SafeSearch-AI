import { organization } from 'redivis';


export default async function getDataset(owner, dataset, { version } = {}) {
	let loadedDataset = null;
	try {
		const response = await organization(dataset.organization?.name || owner).dataset(dataset.name, { version }).get();
		loadedDataset = response;
	} catch (e){
		console.error(e)
	}
	return loadedDataset;
}
