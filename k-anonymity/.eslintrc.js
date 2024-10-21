module.exports = {
	extends: ['eslint:recommended'],
	plugins: [],
	env: {
		node: true,
		es6: true,
	},
	globals: {
		logger: true,
		React: true,
	},
	parserOptions: {
		ecmaVersion: 9,
		sourceType: 'module',
	},
	settings: {},
	rules: {
		// eslint-recommended overrides
		'no-unused-vars': ['warn', { vars: 'all', args: 'none' }], // need to specify args: none
		'no-inner-declarations': 0,

		// Possible errors: https://eslint.org/docs/rules/#possible-errors
		'no-template-curly-in-string': 'warn',
		'no-constant-condition': ['warn', { checkLoops: false }],
		'spaced-comment': ['warn', 'always'],

		// Best practices: https://eslint.org/docs/rules/#possible-errors
		'array-callback-return': 1,
		'dot-notation': 1,
		eqeqeq: 1,
		// 'guard-for-in': 1, // TODO: consider
		'no-caller': 1,
		'no-eval': 1,
		'no-extend-native': 1,
		'no-extra-label': 1,
		'no-implicit-coercion': [1, { boolean: false }],
		'no-implied-eval': 1,
		'no-lone-blocks': 1,
		'no-loop-func': 1,
		'no-iterator': 1,
		'no-multi-str': 1,
		'no-new': 1,
		'no-new-func': 1,
		'no-new-wrappers': 1,
		'no-octal-escape': 1,
		'no-proto': 1,
		'no-return-assign': 1,
		// 'no-return-await': 1, //todo: consider
		'no-script-url': 1,
		'no-self-compare': 1,
		'no-sequences': 1,
		'no-throw-literal': 1,
		'no-unmodified-loop-condition': 1,
		'no-unused-expressions': 1,
		'no-useless-call': 1,
		'no-useless-return': 1,
		'no-void': 1,
		'no-with': 1,
		'prefer-promise-reject-errors': 1,
		'no-warning-comments': [1, { terms: ['todo', 'fixme'] }],
		radix: 1,
		// 'vars-on-top': 1, // TODO: consider
		yoda: 1,

		// Variables: https://eslint.org/docs/rules/#variables
		'no-label-var': 1,
		// 'no-shadow': [1, { builtinGlobals: true, allow: ['err', 'e', 'resolve', 'reject', 'done', 'cb', 'res'] }],
		'no-shadow-restricted-names': 1,
		'no-undef-init': 1,
		'no-use-before-define': [1, { functions: false }],

		// Stylistic
		camelcase: [1, { properties: 'never' }], // we often pass PascalCase properties to server to reflect DB Model references
		'func-style': [1, 'declaration', { allowArrowFunctions: true }],
		'max-depth': [1, { max: 8 }],
		'max-nested-callbacks': [1, { max: 4 }],
		'max-params': [1, { max: 4 }],
		'new-cap': [
			1,
			{
				properties: false,
				capIsNew: false,
			},
		],
		'no-array-constructor': 1,
		// 'no-lonely-if': 1,
		'no-multi-assign': 1,
		'no-negated-condition': 1,
		'no-nested-ternary': 1,
		'no-new-object': 1,
		'one-var': [1, 'never'],

		// ECMA 6: https://eslint.org/docs/rules/#ecmascript-6
		'no-confusing-arrow': 1,
		'no-duplicate-imports': 1,
		'no-useless-computed-key': 1,
		'no-useless-constructor': 1,
		'no-useless-rename': 1,
		'no-var': 1,
		'object-shorthand': ['warn', 'properties'],
		'prefer-arrow-callback': 1,
		'prefer-const': [
			'warn',
			{
				destructuring: 'all',
				ignoreReadBeforeAssign: false,
			},
		],
		// 'prefer-destructuring': [1, { object: true, array: false }], // TODO: enable
		'prefer-numeric-literals': 1,
		'prefer-rest-params': 1,
		'prefer-spread': 1,
		'prefer-template': 1,
		'require-atomic-updates': 0,
	},
};
