import React, { useCallback } from 'react';

import { RedivisButtonText } from '../helpers/Buttons';

import * as styles from './styles.css';

export default function Header({ title }) {
	const renderHeaderLogo = useCallback(() => {
		let headerLogo = (
			<div className={styles.headerLogo} style={{ height: 33, width: 194, backgroundImage: `url(/k-anonymity/assets/header.svg)` }} />
		)
		if (title){
			headerLogo = <a href={`https://labs.redivis.com`}>{headerLogo}</a>
		}
		return headerLogo;
	}, [title]);

	return (
		<div className={styles.headerWrapper}>
			<div className={styles.header}>
				<div className={styles.titleWrapper}>
					{renderHeaderLogo()}
					{title && (
						<>
							<div className={styles.divider} />
							<span>{title}</span>
						</>
					)}
				</div>
				<div className={styles.linkWrapper}>
					<div className={styles.buttonWrapper}>
						<RedivisButtonText
							href={`https://github.com/redivis/k-anonymity`}
							target={'_blank'}
						>
							{'Github'}
						</RedivisButtonText>
					</div>
					<div className={styles.buttonWrapper}>
						<RedivisButtonText
							href={`https://redivis.com`}
							target={'_blank'}
						>
							{'Redivis'}
						</RedivisButtonText>
					</div>
				</div>
			</div>
		</div>
	);
}