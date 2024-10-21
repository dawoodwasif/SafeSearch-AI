import React from 'react';
import { render as reactDOMRender } from 'react-dom';
import { BrowserRouter } from 'react-router-dom';
// Global styles
import './styles.css';

import App from './App';

const root = document.getElementById('root');

reactDOMRender(
	<BrowserRouter>
		<App />
	</BrowserRouter>,
	root,
);
