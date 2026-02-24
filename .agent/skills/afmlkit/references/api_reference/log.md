# API Reference: log.py

**Language**: Python

**Source**: `utils\log.py`

---

## Functions

### setup_logging()

Sets up logging configuration with separate console and file handlers.
Console logs go to stdout while file logs are stored in the directory specified by `LOG_FILE_PATH`.

**Returns**: (none)



### get_logger(name: str)

Returns a logger for the given module name, ensuring logging is configured.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| name | str | - | - |

**Returns**: (none)


