from dotenv import load_dotenv

from settings import Settings


def test_settings_load_from_test_env():
    load_dotenv("config/.env.test", override=True)

    settings = Settings()

    assert settings.ENVIRONMENT == "test"
    assert settings.APP_NAME == "ML App Test"
    assert settings.API_KEY == "test-api-key"
    assert settings.DB_PASSWORD == "test-db-password"
