import React from 'react';

import Button from '@mui/material/Button'
import LoadingButton from '@mui/lab/LoadingButton';

const textStyle = {
	padding: 0,
	textTransform: 'none',
	// color: 'primary.main', // TODO not working, though theme is defined in
	color: '#7f3b97',
	fontWeight: 600,
	fontSize: 14,
	'&:hover': {
		bgcolor: 'unset',
		boxShadow: 'none',
		textDecoration: 'underline',
	},
	'&:active': {
		bgcolor: 'unset',
		boxShadow: 'none',
		textDecoration: 'underline',
	},
};

const secondaryStyle = {
	padding: '5px 10px',
	textTransform: 'none',
	boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.2)',
	bgcolor: 'unset',
	color: '#4d4c54',
	fontSize: 14,
	'&:hover': {
		boxShadow: 'inset 0px 1px 3px rgba(0, 0, 0, 0.2);',
		bgcolor: 'unset',
	},
};

const primaryStyle = {
	padding: '5px 10px',
	textTransform: 'none',
	color: '#ffffff',
	bgcolor: '#7f3b97',
	width: '100%',
	boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.2)',
	fontSize: 14,
	'&:hover': {
		bgcolor: '#7f3b97',
		boxShadow: 'inset 2px 2px 3px #5e1e75',
	},
	'&:disabled': {
		bgcolor: '#f5f5f7',
		color: '#b1b0b8',
		border: '1px solid #d4d3db',
		boxShadow: 'none',
	},
	'&.MuiLoadingButton-loading': {
		color: '#f5f5f7',
	}
}

export function RedivisButton({ type = 'primary', hasLoading = false, ...rest }){
	if (type === 'primary'){
		return hasLoading ? <LoadingButton sx={primaryStyle} {...rest}/> : <Button sx={primaryStyle} {...rest}/>
	} else if (type === 'secondary'){
		return <Button sx={secondaryStyle} {...rest}/>;
	}
	return null;
}

export function RedivisButtonText({ type = 'primary', ...rest }){
	if (type === 'primary'){
		return <Button sx={textStyle} {...rest} />
	}
	return null;
}

