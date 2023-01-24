const provider = new firebase.auth.GoogleAuthProvider();

const btn_content = document.getElementById("button-content");
const btn_loader = document.getElementById("button-loader");

btn_loader.style.display = "none";

function signIn() {
	btn_content.style.display = "none";
	btn_loader.style.display = "block";
	firebase
		.auth()
		.signInWithPopup(provider)
		.then((result) => {
			/** @type {firebase.auth.OAuthCredential} */
			const user = result.user;
			const uid = user.uid;
			window.location = `/token/${uid}`;
		})
		.catch((error) => {
			console.log(error);
			btn_content.style.display = "block";
			btn_loader.style.display = "none";
		});
}
