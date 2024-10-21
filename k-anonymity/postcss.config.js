module.exports = {
	plugins: [
		require('postcss-preset-env')({
			stage: 1,
		}),
		require('postcss-pxtorem')({
			rootValue: 16,
			unitPrecision: 5,
			propList: ['*'],
		}),
	],
};
